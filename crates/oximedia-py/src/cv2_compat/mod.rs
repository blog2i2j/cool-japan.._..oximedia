//! OpenCV-compatible Python API (`oximedia.cv2` submodule).
//!
//! Provides a `cv2`-shaped Python namespace implementing the most commonly
//! used OpenCV functions using OxiMedia's native CV crates.
//!
//! # Python usage
//!
//! ```python
//! from oximedia import cv2
//!
//! img = cv2.imread("photo.png")
//! gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
//! cv2.imwrite("gray.png", gray)
//! ```

pub mod arithmetic;
pub mod color;
pub mod constants;
pub mod contours;
pub mod drawing;
pub mod edges;
pub mod features;
pub mod filters;
pub mod geometry;
pub mod hough;
pub mod image_io;
pub mod morphology;
pub(crate) mod numpy_bridge;
pub mod optical_flow;
pub mod threshold;
pub mod video_capture;
pub mod video_writer;

use pyo3::prelude::*;

/// Register all `cv2`-compatible items into the given Python submodule.
pub fn register_cv2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Constants
    constants::add_constants(m)?;

    // Image I/O
    m.add_function(wrap_pyfunction!(image_io::imread, m)?)?;
    m.add_function(wrap_pyfunction!(image_io::imwrite, m)?)?;
    m.add_function(wrap_pyfunction!(image_io::imdecode, m)?)?;
    m.add_function(wrap_pyfunction!(image_io::imencode, m)?)?;

    // Color conversion
    m.add_function(wrap_pyfunction!(color::cvt_color, m)?)?;

    // Geometry
    m.add_function(wrap_pyfunction!(geometry::resize, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::flip, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::rotate, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::warp_affine, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::get_rotation_matrix_2d, m)?)?;

    // Filters
    m.add_function(wrap_pyfunction!(filters::gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(filters::median_blur, m)?)?;
    m.add_function(wrap_pyfunction!(filters::bilateral_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::filter_2d, m)?)?;
    m.add_function(wrap_pyfunction!(filters::box_filter, m)?)?;

    // Edges
    m.add_function(wrap_pyfunction!(edges::canny, m)?)?;
    m.add_function(wrap_pyfunction!(edges::sobel, m)?)?;
    m.add_function(wrap_pyfunction!(edges::laplacian, m)?)?;

    // Threshold
    m.add_function(wrap_pyfunction!(threshold::threshold, m)?)?;
    m.add_function(wrap_pyfunction!(threshold::adaptive_threshold, m)?)?;

    // Morphology
    m.add_function(wrap_pyfunction!(morphology::erode, m)?)?;
    m.add_function(wrap_pyfunction!(morphology::dilate, m)?)?;
    m.add_function(wrap_pyfunction!(morphology::morphology_ex, m)?)?;
    m.add_function(wrap_pyfunction!(morphology::get_structuring_element, m)?)?;

    // Contours
    m.add_function(wrap_pyfunction!(contours::find_contours, m)?)?;
    m.add_function(wrap_pyfunction!(contours::draw_contours, m)?)?;
    m.add_function(wrap_pyfunction!(contours::contour_area, m)?)?;
    m.add_function(wrap_pyfunction!(contours::bounding_rect, m)?)?;
    m.add_function(wrap_pyfunction!(contours::arc_length, m)?)?;
    m.add_function(wrap_pyfunction!(contours::approx_poly_dp, m)?)?;

    // Drawing
    m.add_function(wrap_pyfunction!(drawing::rectangle, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::circle, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::line, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::put_text, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::polylines, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::fill_poly, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::ellipse, m)?)?;

    // Features
    m.add_function(wrap_pyfunction!(features::good_features_to_track, m)?)?;
    m.add_class::<features::PyORB>()?;
    m.add_function(wrap_pyfunction!(features::orb_create, m)?)?;
    m.add_class::<features::PyKeyPoint>()?;

    // Optical flow
    m.add_function(wrap_pyfunction!(optical_flow::calc_optical_flow_pyr_lk, m)?)?;

    // VideoCapture
    m.add_class::<video_capture::PyVideoCapture>()?;

    // VideoWriter
    m.add_class::<video_writer::PyVideoWriter>()?;
    m.add_function(wrap_pyfunction!(video_writer::video_writer_fourcc, m)?)?;

    // Arithmetic & blending
    m.add_function(wrap_pyfunction!(arithmetic::add_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic::abs_diff, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic::bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic::bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic::bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic::bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic::normalize, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic::in_range, m)?)?;

    // Histogram
    m.add_function(wrap_pyfunction!(arithmetic::calc_hist, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic::equalize_hist, m)?)?;

    // Template matching & min/max location
    m.add_function(wrap_pyfunction!(arithmetic::match_template, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic::min_max_loc, m)?)?;

    // Connected components
    m.add_function(wrap_pyfunction!(arithmetic::connected_components, m)?)?;

    // Hough transforms
    m.add_function(wrap_pyfunction!(hough::hough_lines, m)?)?;
    m.add_function(wrap_pyfunction!(hough::hough_lines_p, m)?)?;
    m.add_function(wrap_pyfunction!(hough::hough_circles, m)?)?;

    Ok(())
}
