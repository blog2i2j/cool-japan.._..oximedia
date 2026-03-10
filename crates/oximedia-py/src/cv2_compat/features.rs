//! Feature detection: goodFeaturesToTrack, ORB, KeyPoint.

use super::image_io::extract_img;
use pyo3::prelude::*;

/// A keypoint as returned by feature detectors.
///
/// Mirrors `cv2.KeyPoint`.
#[pyclass(name = "KeyPoint")]
#[derive(Clone)]
pub struct PyKeyPoint {
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
    #[pyo3(get, set)]
    pub size: f32,
    #[pyo3(get, set)]
    pub angle: f32,
    #[pyo3(get, set)]
    pub response: f32,
    #[pyo3(get, set)]
    pub octave: i32,
    #[pyo3(get, set)]
    pub class_id: i32,
}

#[pymethods]
impl PyKeyPoint {
    #[new]
    #[pyo3(signature = (x=0.0, y=0.0, size=1.0, angle=-1.0, response=0.0, octave=0, class_id=-1))]
    pub fn new(
        x: f32,
        y: f32,
        size: f32,
        angle: f32,
        response: f32,
        octave: i32,
        class_id: i32,
    ) -> Self {
        Self {
            x,
            y,
            size,
            angle,
            response,
            octave,
            class_id,
        }
    }

    /// Return the `pt` attribute (x, y) tuple for OpenCV compatibility.
    #[getter]
    pub fn pt(&self) -> (f32, f32) {
        (self.x, self.y)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "KeyPoint(x={:.1}, y={:.1}, size={:.1}, angle={:.1}, response={:.4})",
            self.x, self.y, self.size, self.angle, self.response
        )
    }
}

/// ORB (Oriented FAST and Rotated BRIEF) feature detector.
///
/// Mirrors `cv2.ORB_create(nfeatures=500, ...)`.
#[pyclass(name = "ORB")]
pub struct PyORB {
    n_features: usize,
    scale_factor: f32,
    n_levels: i32,
    edge_threshold: i32,
    first_level: i32,
    wta_k: i32,
    score_type: i32,
    patch_size: i32,
    fast_threshold: i32,
}

#[pymethods]
impl PyORB {
    #[new]
    #[pyo3(signature = (
        n_features=500,
        scale_factor=1.2,
        n_levels=8,
        edge_threshold=31,
        first_level=0,
        wta_k=2,
        score_type=0,
        patch_size=31,
        fast_threshold=20,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_features: usize,
        scale_factor: f32,
        n_levels: i32,
        edge_threshold: i32,
        first_level: i32,
        wta_k: i32,
        score_type: i32,
        patch_size: i32,
        fast_threshold: i32,
    ) -> Self {
        Self {
            n_features,
            scale_factor,
            n_levels,
            edge_threshold,
            first_level,
            wta_k,
            score_type,
            patch_size,
            fast_threshold,
        }
    }

    /// Detect keypoints in an image.
    pub fn detect(
        &self,
        py: Python<'_>,
        img: Py<PyAny>,
        mask: Option<Py<PyAny>>,
    ) -> PyResult<Vec<PyKeyPoint>> {
        let _ = mask;
        let (data, h, w, ch) = extract_img(py, &img)?;
        let gray = to_gray(&data, w, h, ch);
        let kps = detect_fast_keypoints(
            &gray,
            w,
            h,
            self.fast_threshold as u8,
            self.n_features,
            self.edge_threshold as usize,
        );
        Ok(kps)
    }

    /// Compute descriptors for detected keypoints.
    /// Returns (keypoints, descriptors) — descriptors are returned as None (placeholder).
    pub fn compute(
        &self,
        py: Python<'_>,
        img: Py<PyAny>,
        keypoints: Vec<PyKeyPoint>,
    ) -> PyResult<(Vec<PyKeyPoint>, Py<PyAny>)> {
        let _ = img;
        Ok((keypoints, py.None()))
    }

    /// Detect and compute in one step.
    pub fn detect_and_compute(
        &self,
        py: Python<'_>,
        img: Py<PyAny>,
        mask: Option<Py<PyAny>>,
    ) -> PyResult<(Vec<PyKeyPoint>, Py<PyAny>)> {
        let kps = self.detect(py, img, mask)?;
        Ok((kps, py.None()))
    }

    #[getter]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    #[getter]
    pub fn scale_factor(&self) -> f32 {
        self.scale_factor
    }
    #[getter]
    pub fn n_levels(&self) -> i32 {
        self.n_levels
    }
    #[getter]
    pub fn edge_threshold(&self) -> i32 {
        self.edge_threshold
    }
    #[getter]
    pub fn first_level(&self) -> i32 {
        self.first_level
    }
    #[getter]
    pub fn wta_k(&self) -> i32 {
        self.wta_k
    }
    #[getter]
    pub fn score_type(&self) -> i32 {
        self.score_type
    }
    #[getter]
    pub fn patch_size(&self) -> i32 {
        self.patch_size
    }
    #[getter]
    pub fn fast_threshold(&self) -> i32 {
        self.fast_threshold
    }
}

/// Create an ORB feature detector.
///
/// Mirrors `cv2.ORB_create(nfeatures=500, ...)`.
#[pyfunction]
#[pyo3(name = "ORB_create", signature = (
    n_features=500,
    scale_factor=1.2,
    n_levels=8,
    edge_threshold=31,
    first_level=0,
    wta_k=2,
    score_type=0,
    patch_size=31,
    fast_threshold=20,
))]
#[allow(clippy::too_many_arguments)]
pub fn orb_create(
    n_features: usize,
    scale_factor: f32,
    n_levels: i32,
    edge_threshold: i32,
    first_level: i32,
    wta_k: i32,
    score_type: i32,
    patch_size: i32,
    fast_threshold: i32,
) -> PyResult<PyORB> {
    Ok(PyORB::new(
        n_features,
        scale_factor,
        n_levels,
        edge_threshold,
        first_level,
        wta_k,
        score_type,
        patch_size,
        fast_threshold,
    ))
}

/// Detect corners using the Shi-Tomasi (Good Features to Track) method.
///
/// Mirrors `cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, ...)`.
/// Returns a list of (x, y) float tuples.
#[pyfunction]
#[pyo3(name = "goodFeaturesToTrack", signature = (
    image,
    max_corners,
    quality_level,
    min_distance,
    mask=None,
    block_size=3,
    use_harris_detector=false,
    k=0.04,
))]
#[allow(clippy::too_many_arguments)]
pub fn good_features_to_track(
    py: Python<'_>,
    image: Py<PyAny>,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
    mask: Option<Py<PyAny>>,
    block_size: usize,
    use_harris_detector: bool,
    k: f64,
) -> PyResult<Py<PyAny>> {
    let _ = (mask, use_harris_detector);
    let (data, h, w, ch) = extract_img(py, &image)?;
    let gray = to_gray(&data, w, h, ch);

    // Compute corner response using Harris/Shi-Tomasi
    let corners = if use_harris_detector {
        compute_harris_response(&gray, w, h, block_size, k)
    } else {
        compute_shi_tomasi_response(&gray, w, h, block_size)
    };

    // Find max response for quality threshold
    let max_resp = corners.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max_resp * quality_level as f32;

    // Collect candidate corners above threshold
    let mut candidates: Vec<(usize, usize, f32)> = corners
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r >= threshold)
        .map(|(idx, &r)| (idx % w, idx / w, r))
        .collect();

    // Sort by response (descending)
    candidates.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Non-maximum suppression by min_distance
    let mut selected: Vec<(f32, f32)> = Vec::new();
    for (cx, cy, _) in candidates {
        if selected.len() >= max_corners {
            break;
        }
        let too_close = selected.iter().any(|&(sx, sy)| {
            let dx = cx as f32 - sx;
            let dy = cy as f32 - sy;
            ((dx * dx + dy * dy) as f64).sqrt() < min_distance
        });
        if !too_close {
            selected.push((cx as f32, cy as f32));
        }
    }

    // Return as list of (x, y) float tuples
    let result = pyo3::types::PyList::empty(py);
    for (x, y) in selected {
        result.append(pyo3::types::PyTuple::new(py, [x, y])?)?;
    }
    Ok(result.into())
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn to_gray(data: &[u8], w: usize, h: usize, ch: usize) -> Vec<f32> {
    let mut gray = vec![0.0f32; w * h];
    if ch == 1 {
        for (i, &v) in data.iter().enumerate() {
            gray[i] = v as f32;
        }
    } else {
        for i in 0..w * h {
            let off = i * ch;
            // BGR order
            gray[i] = 0.114 * data[off] as f32
                + 0.587 * data[off + 1] as f32
                + 0.299 * data[off + 2] as f32;
        }
    }
    gray
}

fn compute_harris_response(gray: &[f32], w: usize, h: usize, block: usize, k: f64) -> Vec<f32> {
    let k = k as f32;
    // Sobel gradients
    let (gx, gy) = sobel_gradients(gray, w, h);
    let mut response = vec![0.0f32; w * h];
    let half = block / 2;

    for y in half..h.saturating_sub(half) {
        for x in half..w.saturating_sub(half) {
            let mut ixx = 0.0f32;
            let mut iyy = 0.0f32;
            let mut ixy = 0.0f32;
            for dy in 0..block {
                for dx in 0..block {
                    let ny = y + dy - half;
                    let nx = x + dx - half;
                    let gi = ny * w + nx;
                    ixx += gx[gi] * gx[gi];
                    iyy += gy[gi] * gy[gi];
                    ixy += gx[gi] * gy[gi];
                }
            }
            let det = ixx * iyy - ixy * ixy;
            let trace = ixx + iyy;
            response[y * w + x] = det - k * trace * trace;
        }
    }
    response
}

fn compute_shi_tomasi_response(gray: &[f32], w: usize, h: usize, block: usize) -> Vec<f32> {
    let (gx, gy) = sobel_gradients(gray, w, h);
    let mut response = vec![0.0f32; w * h];
    let half = block / 2;

    for y in half..h.saturating_sub(half) {
        for x in half..w.saturating_sub(half) {
            let mut ixx = 0.0f32;
            let mut iyy = 0.0f32;
            let mut ixy = 0.0f32;
            for dy in 0..block {
                for dx in 0..block {
                    let ny = y + dy - half;
                    let nx = x + dx - half;
                    let gi = ny * w + nx;
                    ixx += gx[gi] * gx[gi];
                    iyy += gy[gi] * gy[gi];
                    ixy += gx[gi] * gy[gi];
                }
            }
            // Minimum eigenvalue of M = [[ixx, ixy],[ixy, iyy]]
            let trace = ixx + iyy;
            let det = ixx * iyy - ixy * ixy;
            let disc = ((trace * trace / 4.0) - det).max(0.0).sqrt();
            let lambda_min = trace / 2.0 - disc;
            response[y * w + x] = lambda_min.max(0.0);
        }
    }
    response
}

fn sobel_gradients(gray: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let get = |dy: isize, dx: isize| {
                gray[((y as isize + dy) as usize) * w + (x as isize + dx) as usize]
            };
            gx[y * w + x] = -get(-1, -1) + get(-1, 1) - 2.0 * get(0, -1) + 2.0 * get(0, 1)
                - get(1, -1)
                + get(1, 1);
            gy[y * w + x] = -get(-1, -1) - 2.0 * get(-1, 0) - get(-1, 1)
                + get(1, -1)
                + 2.0 * get(1, 0)
                + get(1, 1);
        }
    }
    (gx, gy)
}

fn detect_fast_keypoints(
    gray: &[f32],
    w: usize,
    h: usize,
    threshold: u8,
    max_kp: usize,
    edge: usize,
) -> Vec<PyKeyPoint> {
    let t = threshold as f32;
    let mut kps = Vec::new();

    // FAST circle offsets (radius 3)
    let circle: [(isize, isize); 16] = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ];

    for y in edge..h.saturating_sub(edge) {
        for x in edge..w.saturating_sub(edge) {
            let center = gray[y * w + x];
            // Quick test: check 4 cardinal points
            let test_4 = [0usize, 4, 8, 12]
                .iter()
                .filter(|&&i| {
                    let (dx, dy) = circle[i];
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    (gray[ny * w + nx] - center).abs() > t
                })
                .count();

            if test_4 < 3 {
                continue;
            }

            // Full FAST-9 test
            let mut n_brighter = 0u32;
            let mut n_darker = 0u32;
            for &(dx, dy) in &circle {
                let nx = (x as isize + dx) as usize;
                let ny = (y as isize + dy) as usize;
                let diff = gray[ny * w + nx] - center;
                if diff > t {
                    n_brighter += 1;
                } else if diff < -t {
                    n_darker += 1;
                }
            }

            if n_brighter >= 9 || n_darker >= 9 {
                let response = (n_brighter.max(n_darker) as f32) * t;
                kps.push(PyKeyPoint::new(
                    x as f32, y as f32, 7.0, -1.0, response, 0, -1,
                ));
                if kps.len() >= max_kp * 2 {
                    break;
                }
            }
        }
        if kps.len() >= max_kp * 2 {
            break;
        }
    }

    // Sort by response and truncate
    kps.sort_unstable_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    kps.truncate(max_kp);
    kps
}
