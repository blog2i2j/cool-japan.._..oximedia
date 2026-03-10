//! OpenCV-compatible constants for `oximedia.cv2`.

use pyo3::prelude::*;

pub fn add_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // imread flags
    m.add("IMREAD_COLOR", 1i32)?;
    m.add("IMREAD_GRAYSCALE", 0i32)?;
    m.add("IMREAD_UNCHANGED", -1i32)?;
    m.add("IMREAD_ANYDEPTH", 2i32)?;
    m.add("IMREAD_ANYCOLOR", 4i32)?;

    // Color conversion codes
    m.add("COLOR_BGR2BGRA", 0i32)?;
    m.add("COLOR_BGRA2BGR", 1i32)?;
    m.add("COLOR_BGR2RGBA", 2i32)?;
    m.add("COLOR_RGBA2BGR", 3i32)?;
    m.add("COLOR_BGR2RGB", 4i32)?;
    m.add("COLOR_RGB2BGR", 4i32)?;
    m.add("COLOR_BGRA2RGBA", 5i32)?;
    m.add("COLOR_BGR2GRAY", 6i32)?;
    m.add("COLOR_RGB2GRAY", 7i32)?;
    m.add("COLOR_GRAY2BGR", 8i32)?;
    m.add("COLOR_GRAY2RGB", 8i32)?;
    m.add("COLOR_GRAY2BGRA", 9i32)?;
    m.add("COLOR_BGRA2GRAY", 10i32)?;
    m.add("COLOR_BGR2HSV", 40i32)?;
    m.add("COLOR_RGB2HSV", 41i32)?;
    m.add("COLOR_BGR2Lab", 44i32)?;
    m.add("COLOR_RGB2Lab", 45i32)?;
    m.add("COLOR_BGR2HLS", 52i32)?;
    m.add("COLOR_BGR2YUV", 82i32)?;
    m.add("COLOR_RGB2YUV", 83i32)?;
    m.add("COLOR_YUV2BGR", 84i32)?;
    m.add("COLOR_YUV2RGB", 85i32)?;
    m.add("COLOR_YUV2GRAY_420", 106i32)?;
    m.add("COLOR_YUV2BGR_NV12", 90i32)?;
    m.add("COLOR_Lab2BGR", 56i32)?;
    m.add("COLOR_Lab2RGB", 57i32)?;
    m.add("COLOR_HSV2BGR", 54i32)?;
    m.add("COLOR_HSV2RGB", 55i32)?;
    m.add("COLOR_HLS2BGR", 60i32)?;
    m.add("COLOR_HLS2RGB", 61i32)?;

    // Interpolation flags
    m.add("INTER_NEAREST", 0i32)?;
    m.add("INTER_LINEAR", 1i32)?;
    m.add("INTER_CUBIC", 2i32)?;
    m.add("INTER_AREA", 3i32)?;
    m.add("INTER_LANCZOS4", 4i32)?;
    m.add("INTER_LINEAR_EXACT", 5i32)?;

    // Border types
    m.add("BORDER_CONSTANT", 0i32)?;
    m.add("BORDER_REPLICATE", 1i32)?;
    m.add("BORDER_REFLECT", 4i32)?;
    m.add("BORDER_WRAP", 3i32)?;
    m.add("BORDER_REFLECT_101", 4i32)?;
    m.add("BORDER_TRANSPARENT", 5i32)?;
    m.add("BORDER_DEFAULT", 4i32)?;

    // Threshold types
    m.add("THRESH_BINARY", 0i32)?;
    m.add("THRESH_BINARY_INV", 1i32)?;
    m.add("THRESH_TRUNC", 2i32)?;
    m.add("THRESH_TOZERO", 3i32)?;
    m.add("THRESH_TOZERO_INV", 4i32)?;
    m.add("THRESH_MASK", 7i32)?;
    m.add("THRESH_OTSU", 8i32)?;
    m.add("THRESH_TRIANGLE", 16i32)?;

    // Adaptive threshold methods
    m.add("ADAPTIVE_THRESH_MEAN_C", 0i32)?;
    m.add("ADAPTIVE_THRESH_GAUSSIAN_C", 1i32)?;

    // Morphology shapes
    m.add("MORPH_RECT", 0i32)?;
    m.add("MORPH_CROSS", 1i32)?;
    m.add("MORPH_ELLIPSE", 2i32)?;

    // Morphology operations
    m.add("MORPH_ERODE", 0i32)?;
    m.add("MORPH_DILATE", 1i32)?;
    m.add("MORPH_OPEN", 2i32)?;
    m.add("MORPH_CLOSE", 3i32)?;
    m.add("MORPH_GRADIENT", 4i32)?;
    m.add("MORPH_TOPHAT", 5i32)?;
    m.add("MORPH_BLACKHAT", 6i32)?;

    // Contour retrieval modes
    m.add("RETR_EXTERNAL", 0i32)?;
    m.add("RETR_LIST", 1i32)?;
    m.add("RETR_CCOMP", 2i32)?;
    m.add("RETR_TREE", 3i32)?;

    // Contour approximation methods
    m.add("CHAIN_APPROX_NONE", 1i32)?;
    m.add("CHAIN_APPROX_SIMPLE", 2i32)?;
    m.add("CHAIN_APPROX_TC89_L1", 3i32)?;
    m.add("CHAIN_APPROX_TC89_KCOS", 4i32)?;

    // VideoCapture properties
    m.add("CAP_PROP_POS_MSEC", 0i32)?;
    m.add("CAP_PROP_POS_FRAMES", 1i32)?;
    m.add("CAP_PROP_POS_AVI_RATIO", 2i32)?;
    m.add("CAP_PROP_FRAME_WIDTH", 3i32)?;
    m.add("CAP_PROP_FRAME_HEIGHT", 4i32)?;
    m.add("CAP_PROP_FPS", 5i32)?;
    m.add("CAP_PROP_FOURCC", 6i32)?;
    m.add("CAP_PROP_FRAME_COUNT", 7i32)?;
    m.add("CAP_PROP_FORMAT", 8i32)?;
    m.add("CAP_PROP_MODE", 9i32)?;

    // Rotation codes
    m.add("ROTATE_90_CLOCKWISE", 0i32)?;
    m.add("ROTATE_180", 1i32)?;
    m.add("ROTATE_90_COUNTERCLOCKWISE", 2i32)?;

    // Font types
    m.add("FONT_HERSHEY_SIMPLEX", 0i32)?;
    m.add("FONT_HERSHEY_PLAIN", 1i32)?;
    m.add("FONT_HERSHEY_DUPLEX", 2i32)?;
    m.add("FONT_HERSHEY_COMPLEX", 3i32)?;
    m.add("FONT_HERSHEY_TRIPLEX", 4i32)?;
    m.add("FONT_HERSHEY_COMPLEX_SMALL", 5i32)?;
    m.add("FONT_HERSHEY_SCRIPT_SIMPLEX", 6i32)?;
    m.add("FONT_HERSHEY_SCRIPT_COMPLEX", 7i32)?;
    m.add("FONT_ITALIC", 16i32)?;

    // Line types
    m.add("LINE_4", 4i32)?;
    m.add("LINE_8", 8i32)?;
    m.add("LINE_AA", 16i32)?;
    m.add("FILLED", -1i32)?;

    // Feature detector flags
    m.add("ORB_HARRIS_SCORE", 0i32)?;
    m.add("ORB_FAST_SCORE", 1i32)?;

    // Hough transform
    m.add("HOUGH_STANDARD", 0i32)?;
    m.add("HOUGH_PROBABILISTIC", 1i32)?;

    // Norm types
    m.add("NORM_INF", 1i32)?;
    m.add("NORM_L1", 2i32)?;
    m.add("NORM_L2", 4i32)?;
    m.add("NORM_MINMAX", 32i32)?;

    // Template matching methods
    m.add("TM_SQDIFF", 0i32)?;
    m.add("TM_SQDIFF_NORMED", 1i32)?;
    m.add("TM_CCORR", 2i32)?;
    m.add("TM_CCORR_NORMED", 3i32)?;
    m.add("TM_CCOEFF", 4i32)?;
    m.add("TM_CCOEFF_NORMED", 5i32)?;

    // Comparison operations
    m.add("CMP_EQ", 0i32)?;
    m.add("CMP_GT", 1i32)?;
    m.add("CMP_GE", 2i32)?;
    m.add("CMP_LT", 3i32)?;
    m.add("CMP_LE", 4i32)?;
    m.add("CMP_NE", 5i32)?;

    // Data types (like CV_8U, CV_8S, etc.)
    m.add("CV_8U", 0i32)?;
    m.add("CV_8S", 1i32)?;
    m.add("CV_16U", 2i32)?;
    m.add("CV_16S", 3i32)?;
    m.add("CV_32S", 4i32)?;
    m.add("CV_32F", 5i32)?;
    m.add("CV_64F", 6i32)?;

    // Hough transform methods
    m.add("HOUGH_STANDARD", 0i32)?;
    m.add("HOUGH_PROBABILISTIC", 1i32)?;
    m.add("HOUGH_MULTI_SCALE", 2i32)?;
    m.add("HOUGH_GRADIENT", 3i32)?;
    m.add("HOUGH_GRADIENT_ALT", 4i32)?;

    Ok(())
}
