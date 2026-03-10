//! Peak Signal-to-Noise Ratio (PSNR) calculation.
//!
//! PSNR is one of the most widely used objective quality metrics in video
//! compression and transmission. It measures the ratio between the maximum
//! possible signal power and the power of corrupting noise.
//!
//! # Formula
//!
//! ```text
//! MSE = (1/N) * Σ(reference[i] - distorted[i])²
//! PSNR = 10 * log₁₀(MAX²/MSE)
//! ```
//!
//! Where MAX is the maximum possible pixel value (255 for 8-bit, 1023 for 10-bit, etc.).
//!
//! # Examples
//!
//! ```
//! use oximedia_cv::quality::psnr::calculate_psnr;
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 640, 480);
//! reference.allocate();
//! let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 640, 480);
//! distorted.allocate();
//!
//! let result = calculate_psnr(&reference, &distorted).expect("psnr should succeed");
//! println!("PSNR: {:.2} dB", result);
//! ```

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

/// Result of PSNR calculation including per-plane and overall values.
#[derive(Debug, Clone, PartialEq)]
pub struct PsnrResult {
    /// Overall PSNR value in dB.
    pub overall: f64,

    /// Per-plane PSNR values (Y, U, V or R, G, B).
    pub per_plane: Vec<f64>,

    /// Mean Squared Error values per plane.
    pub mse_per_plane: Vec<f64>,

    /// Overall MSE.
    pub mse_overall: f64,
}

impl PsnrResult {
    /// Create a new PSNR result.
    #[must_use]
    pub fn new(
        overall: f64,
        per_plane: Vec<f64>,
        mse_per_plane: Vec<f64>,
        mse_overall: f64,
    ) -> Self {
        Self {
            overall,
            per_plane,
            mse_per_plane,
            mse_overall,
        }
    }

    /// Check if PSNR indicates acceptable quality (> 30 dB).
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.overall > 30.0
    }

    /// Check if PSNR indicates high quality (> 40 dB).
    #[must_use]
    pub fn is_high_quality(&self) -> bool {
        self.overall > 40.0
    }

    /// Get the luma PSNR (first plane, typically Y component).
    #[must_use]
    pub fn luma_psnr(&self) -> Option<f64> {
        self.per_plane.first().copied()
    }
}

/// Calculate PSNR between reference and distorted frames.
///
/// Returns the overall PSNR value averaged across all planes.
///
/// # Errors
///
/// Returns an error if:
/// - Frame dimensions don't match
/// - Pixel formats are incompatible
/// - Planes have insufficient data
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::psnr::calculate_psnr;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut ref_frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// ref_frame.allocate();
/// let mut dist_frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// dist_frame.allocate();
///
/// let psnr = calculate_psnr(&ref_frame, &dist_frame).expect("psnr should succeed");
/// assert!(psnr >= 0.0);
/// ```
pub fn calculate_psnr(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<f64> {
    let result = calculate_psnr_planes(reference, distorted)?;
    Ok(result.overall)
}

/// Calculate PSNR with detailed per-plane results.
///
/// Returns a `PsnrResult` containing overall PSNR, per-plane PSNR values,
/// and MSE values for detailed analysis.
///
/// # Errors
///
/// Returns an error if frames are incompatible or have mismatched dimensions.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::psnr::calculate_psnr_planes;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// reference.allocate();
/// let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// distorted.allocate();
///
/// let result = calculate_psnr_planes(&reference, &distorted).expect("psnr planes should succeed");
/// println!("Y-PSNR: {:.2} dB", result.per_plane[0]);
/// println!("U-PSNR: {:.2} dB", result.per_plane[1]);
/// println!("V-PSNR: {:.2} dB", result.per_plane[2]);
/// ```
pub fn calculate_psnr_planes(
    reference: &VideoFrame,
    distorted: &VideoFrame,
) -> CvResult<PsnrResult> {
    // Validate inputs
    validate_frames(reference, distorted)?;

    let num_planes = reference.planes.len();
    let mut per_plane_psnr = Vec::with_capacity(num_planes);
    let mut per_plane_mse = Vec::with_capacity(num_planes);

    let bit_depth = get_bit_depth(reference.format);
    let max_value = (1u32 << bit_depth) - 1;

    // Calculate PSNR for each plane
    for plane_idx in 0..num_planes {
        let ref_plane = &reference.planes[plane_idx];
        let dist_plane = &distorted.planes[plane_idx];

        let mse = calculate_mse(&ref_plane.data, &dist_plane.data, bit_depth)?;

        per_plane_mse.push(mse);

        let psnr = if mse < 1e-10 {
            100.0 // Cap at 100 dB for near-identical planes
        } else {
            let max_val_sq = f64::from(max_value) * f64::from(max_value);
            10.0 * (max_val_sq / mse).log10()
        };

        per_plane_psnr.push(psnr);
    }

    // Calculate overall PSNR
    // For YUV formats, weight luma more heavily (6:1:1 ratio)
    let overall_psnr = if num_planes >= 3 {
        let y_psnr = per_plane_psnr[0];
        let u_psnr = per_plane_psnr[1];
        let v_psnr = per_plane_psnr[2];

        // Convert from dB to linear scale for averaging
        let y_mse = 10.0_f64.powf(-y_psnr / 10.0);
        let u_mse = 10.0_f64.powf(-u_psnr / 10.0);
        let v_mse = 10.0_f64.powf(-v_psnr / 10.0);

        // Weighted average (6:1:1)
        let weighted_mse = (6.0 * y_mse + u_mse + v_mse) / 8.0;
        -10.0 * weighted_mse.log10()
    } else {
        // For single-plane or RGB, simple average
        per_plane_psnr.iter().sum::<f64>() / per_plane_psnr.len() as f64
    };

    let overall_mse = per_plane_mse.iter().sum::<f64>() / per_plane_mse.len() as f64;

    Ok(PsnrResult::new(
        overall_psnr,
        per_plane_psnr,
        per_plane_mse,
        overall_mse,
    ))
}

/// Calculate Mean Squared Error between two data buffers.
///
/// Supports different bit depths by normalizing values to 8-bit range.
///
/// # Errors
///
/// Returns an error if buffers have different lengths.
fn calculate_mse(reference: &[u8], distorted: &[u8], bit_depth: u32) -> CvResult<f64> {
    if reference.len() != distorted.len() {
        return Err(CvError::invalid_parameter(
            "buffer_length",
            format!("{} vs {}", reference.len(), distorted.len()),
        ));
    }

    if reference.is_empty() {
        return Ok(0.0);
    }

    let mut sum_squared_error = 0.0;
    let shift = bit_depth.saturating_sub(8);

    if bit_depth <= 8 {
        // 8-bit path
        for (&ref_val, &dist_val) in reference.iter().zip(distorted.iter()) {
            let diff = f64::from(ref_val) - f64::from(dist_val);
            sum_squared_error += diff * diff;
        }
    } else {
        // High bit-depth path (10-bit, 12-bit, etc.)
        // Pixels are stored as 16-bit little-endian values
        let ref_pixels = bytes_to_u16_le(reference);
        let dist_pixels = bytes_to_u16_le(distorted);

        for (&ref_val, &dist_val) in ref_pixels.iter().zip(dist_pixels.iter()) {
            let ref_normalized = (ref_val >> shift) as f64;
            let dist_normalized = (dist_val >> shift) as f64;
            let diff = ref_normalized - dist_normalized;
            sum_squared_error += diff * diff;
        }
    }

    let mse = sum_squared_error / reference.len() as f64;
    Ok(mse)
}

/// Convert byte slice to u16 values (little-endian).
fn bytes_to_u16_le(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

/// Get bit depth for a pixel format.
fn get_bit_depth(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Yuv420p10le => 10,
        PixelFormat::Yuv420p12le | PixelFormat::Gray16 => 12,
        _ => 8, // Default to 8-bit for 8-bit or unknown formats
    }
}

/// Validate that two frames are compatible for comparison.
fn validate_frames(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<()> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(CvError::invalid_parameter(
            "dimensions",
            format!(
                "{}x{} vs {}x{}",
                reference.width, reference.height, distorted.width, distorted.height
            ),
        ));
    }

    if reference.format != distorted.format {
        return Err(CvError::invalid_parameter(
            "pixel_format",
            "Frames must have the same pixel format",
        ));
    }

    if reference.planes.len() != distorted.planes.len() {
        return Err(CvError::invalid_parameter(
            "plane_count",
            format!("{} vs {}", reference.planes.len(), distorted.planes.len()),
        ));
    }

    for (idx, (ref_plane, dist_plane)) in reference
        .planes
        .iter()
        .zip(distorted.planes.iter())
        .enumerate()
    {
        if ref_plane.data.len() != dist_plane.data.len() {
            return Err(CvError::invalid_parameter(
                "plane_size",
                format!(
                    "Plane {}: {} vs {} bytes",
                    idx,
                    ref_plane.data.len(),
                    dist_plane.data.len()
                ),
            ));
        }
    }

    Ok(())
}

/// Calculate PSNR for a specific plane.
///
/// Useful for analyzing quality on specific color components.
///
/// # Errors
///
/// Returns an error if plane index is out of bounds or data is invalid.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::psnr::calculate_plane_psnr;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// reference.allocate();
/// let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// distorted.allocate();
///
/// // Calculate Y-plane PSNR only
/// let y_psnr = calculate_plane_psnr(&reference, &distorted, 0).expect("plane psnr should succeed");
/// println!("Luma PSNR: {:.2} dB", y_psnr);
/// ```
pub fn calculate_plane_psnr(
    reference: &VideoFrame,
    distorted: &VideoFrame,
    plane_index: usize,
) -> CvResult<f64> {
    validate_frames(reference, distorted)?;

    if plane_index >= reference.planes.len() {
        return Err(CvError::invalid_parameter(
            "plane_index",
            format!("{} (max: {})", plane_index, reference.planes.len() - 1),
        ));
    }

    let ref_plane = &reference.planes[plane_index];
    let dist_plane = &distorted.planes[plane_index];

    let bit_depth = get_bit_depth(reference.format);
    let max_value = (1u32 << bit_depth) - 1;

    let mse = calculate_mse(&ref_plane.data, &dist_plane.data, bit_depth)?;

    if mse < 1e-10 {
        Ok(100.0)
    } else {
        let max_val_sq = f64::from(max_value) * f64::from(max_value);
        Ok(10.0 * (max_val_sq / mse).log10())
    }
}

/// Calculate MSE (Mean Squared Error) between two frames.
///
/// Returns the average MSE across all planes.
///
/// # Errors
///
/// Returns an error if frames are incompatible.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::psnr::calculate_frame_mse;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// reference.allocate();
/// let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// distorted.allocate();
///
/// let mse = calculate_frame_mse(&reference, &distorted).expect("frame mse should succeed");
/// assert!(mse >= 0.0);
/// ```
pub fn calculate_frame_mse(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<f64> {
    validate_frames(reference, distorted)?;

    let bit_depth = get_bit_depth(reference.format);
    let mut total_mse = 0.0;

    for (ref_plane, dist_plane) in reference.planes.iter().zip(distorted.planes.iter()) {
        let mse = calculate_mse(&ref_plane.data, &dist_plane.data, bit_depth)?;
        total_mse += mse;
    }

    Ok(total_mse / reference.planes.len() as f64)
}

/// Calculate PSNR between two raw pixel buffers.
///
/// Lower-level function for calculating PSNR on raw data.
///
/// # Errors
///
/// Returns an error if buffers have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::psnr::calculate_buffer_psnr;
///
/// let reference = vec![255u8; 1024];
/// let distorted = vec![250u8; 1024];
///
/// let psnr = calculate_buffer_psnr(&reference, &distorted, 8).expect("buffer psnr should succeed");
/// assert!(psnr > 0.0);
/// ```
pub fn calculate_buffer_psnr(reference: &[u8], distorted: &[u8], bit_depth: u32) -> CvResult<f64> {
    if reference.is_empty() || distorted.is_empty() {
        return Err(CvError::insufficient_data(1, 0));
    }

    if reference.len() != distorted.len() {
        return Err(CvError::invalid_parameter(
            "buffer_length",
            format!("{} vs {}", reference.len(), distorted.len()),
        ));
    }

    let mse = calculate_mse(reference, distorted, bit_depth)?;

    if mse < 1e-10 {
        Ok(100.0)
    } else {
        let max_value = (1u32 << bit_depth) - 1;
        let max_val_sq = f64::from(max_value) * f64::from(max_value);
        Ok(10.0 * (max_val_sq / mse).log10())
    }
}

/// Calculate Signal-to-Noise Ratio (SNR) in dB.
///
/// SNR is similar to PSNR but measures against signal power instead of max value.
///
/// # Errors
///
/// Returns an error if frames are incompatible.
pub fn calculate_snr(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<f64> {
    validate_frames(reference, distorted)?;

    let bit_depth = get_bit_depth(reference.format);
    let mut signal_power = 0.0;
    let mut noise_power = 0.0;
    let mut total_samples = 0;

    for (ref_plane, dist_plane) in reference.planes.iter().zip(distorted.planes.iter()) {
        for (&ref_val, &dist_val) in ref_plane.data.iter().zip(dist_plane.data.iter()) {
            let ref_f = f64::from(ref_val);
            let dist_f = f64::from(dist_val);

            signal_power += ref_f * ref_f;
            let noise = ref_f - dist_f;
            noise_power += noise * noise;
            total_samples += 1;
        }
    }

    if total_samples == 0 {
        return Ok(0.0);
    }

    signal_power /= total_samples as f64;
    noise_power /= total_samples as f64;

    if noise_power < 1e-10 {
        Ok(100.0)
    } else {
        Ok(10.0 * (signal_power / noise_power).log10())
    }
}

/// PSNR quality categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsnrQuality {
    /// Excellent quality (> 45 dB).
    Excellent,
    /// Very good quality (40-45 dB).
    VeryGood,
    /// Good quality (35-40 dB).
    Good,
    /// Acceptable quality (30-35 dB).
    Acceptable,
    /// Poor quality (25-30 dB).
    Poor,
    /// Very poor quality (< 25 dB).
    VeryPoor,
}

impl PsnrQuality {
    /// Categorize PSNR value into quality level.
    #[must_use]
    pub fn from_psnr(psnr: f64) -> Self {
        if psnr >= 45.0 {
            Self::Excellent
        } else if psnr >= 40.0 {
            Self::VeryGood
        } else if psnr >= 35.0 {
            Self::Good
        } else if psnr >= 30.0 {
            Self::Acceptable
        } else if psnr >= 25.0 {
            Self::Poor
        } else {
            Self::VeryPoor
        }
    }

    /// Get descriptive string for quality level.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent - visually lossless",
            Self::VeryGood => "Very Good - imperceptible artifacts",
            Self::Good => "Good - minor artifacts",
            Self::Acceptable => "Acceptable - visible but not annoying",
            Self::Poor => "Poor - annoying artifacts",
            Self::VeryPoor => "Very Poor - severe degradation",
        }
    }
}

/// Calculate weighted PSNR with custom plane weights.
///
/// Allows specifying custom weights for each plane instead of the default 6:1:1 ratio.
///
/// # Errors
///
/// Returns an error if weights length doesn't match plane count.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::psnr::calculate_weighted_psnr;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// reference.allocate();
/// let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// distorted.allocate();
///
/// // Equal weights for all planes
/// let weights = vec![1.0, 1.0, 1.0];
/// let psnr = calculate_weighted_psnr(&reference, &distorted, &weights).expect("weighted psnr should succeed");
/// ```
pub fn calculate_weighted_psnr(
    reference: &VideoFrame,
    distorted: &VideoFrame,
    weights: &[f64],
) -> CvResult<f64> {
    let result = calculate_psnr_planes(reference, distorted)?;

    if result.per_plane.len() != weights.len() {
        return Err(CvError::invalid_parameter(
            "weights",
            format!(
                "Expected {} weights, got {}",
                result.per_plane.len(),
                weights.len()
            ),
        ));
    }

    // Normalize weights
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return Err(CvError::invalid_parameter(
            "weights",
            "Sum must be positive",
        ));
    }

    // Convert PSNR to linear scale, apply weights, convert back
    let mut weighted_mse = 0.0;
    for (psnr, weight) in result.per_plane.iter().zip(weights.iter()) {
        let mse = 10.0_f64.powf(-psnr / 10.0);
        weighted_mse += weight * mse;
    }
    weighted_mse /= total_weight;

    Ok(-10.0 * weighted_mse.log10())
}

/// Calculate PSNR for a region of interest.
///
/// Computes PSNR only within the specified rectangular region.
///
/// # Errors
///
/// Returns an error if ROI is out of bounds.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::psnr::calculate_roi_psnr;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// reference.allocate();
/// let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// distorted.allocate();
///
/// // Calculate PSNR for center region
/// let psnr = calculate_roi_psnr(&reference, &distorted, 480, 270, 960, 540).expect("roi psnr should succeed");
/// ```
pub fn calculate_roi_psnr(
    reference: &VideoFrame,
    distorted: &VideoFrame,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> CvResult<f64> {
    validate_frames(reference, distorted)?;

    // Validate ROI
    if x + width > reference.width || y + height > reference.height {
        return Err(CvError::invalid_roi(x, y, width, height));
    }

    let bit_depth = get_bit_depth(reference.format);
    let max_value = (1u32 << bit_depth) - 1;

    let mut total_mse = 0.0;
    let num_planes = reference.planes.len();

    for plane_idx in 0..num_planes {
        let ref_plane = &reference.planes[plane_idx];
        let dist_plane = &distorted.planes[plane_idx];

        let (plane_width, plane_height) = reference.plane_dimensions(plane_idx);
        let (h_ratio, v_ratio) = reference.format.chroma_subsampling();

        // Adjust ROI for chroma subsampling
        let plane_x = if plane_idx == 0 { x } else { x / h_ratio };
        let plane_y = if plane_idx == 0 { y } else { y / v_ratio };
        let plane_roi_width = if plane_idx == 0 {
            width
        } else {
            width / h_ratio
        };
        let plane_roi_height = if plane_idx == 0 {
            height
        } else {
            height / v_ratio
        };

        // Calculate MSE for ROI
        let mut mse_sum = 0.0;
        let mut pixel_count = 0;

        for py in plane_y..plane_y + plane_roi_height {
            for px in plane_x..plane_x + plane_roi_width {
                if py < plane_height && px < plane_width {
                    let idx = (py * ref_plane.stride as u32 + px) as usize;
                    if idx < ref_plane.data.len() && idx < dist_plane.data.len() {
                        let diff = f64::from(ref_plane.data[idx]) - f64::from(dist_plane.data[idx]);
                        mse_sum += diff * diff;
                        pixel_count += 1;
                    }
                }
            }
        }

        if pixel_count > 0 {
            total_mse += mse_sum / pixel_count as f64;
        }
    }

    let avg_mse = total_mse / num_planes as f64;

    if avg_mse < 1e-10 {
        Ok(100.0)
    } else {
        let max_val_sq = f64::from(max_value) * f64::from(max_value);
        Ok(10.0 * (max_val_sq / avg_mse).log10())
    }
}

/// Calculate block-wise PSNR map.
///
/// Returns a 2D array of PSNR values for each block in the frame.
/// Useful for identifying spatial distribution of quality.
///
/// # Errors
///
/// Returns an error if frames are incompatible.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::psnr::calculate_psnr_map;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// reference.allocate();
/// let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// distorted.allocate();
///
/// let psnr_map = calculate_psnr_map(&reference, &distorted, 64).expect("psnr map should succeed");
/// // psnr_map[y][x] contains PSNR for block at (x, y)
/// ```
pub fn calculate_psnr_map(
    reference: &VideoFrame,
    distorted: &VideoFrame,
    block_size: usize,
) -> CvResult<Vec<Vec<f64>>> {
    validate_frames(reference, distorted)?;

    let width = reference.width as usize;
    let height = reference.height as usize;
    let blocks_x = width.div_ceil(block_size);
    let blocks_y = height.div_ceil(block_size);

    let mut psnr_map = vec![vec![0.0; blocks_x]; blocks_y];

    let bit_depth = get_bit_depth(reference.format);
    let max_value = (1u32 << bit_depth) - 1;
    let max_val_sq = f64::from(max_value) * f64::from(max_value);

    // Work on luma plane
    let ref_plane = &reference.planes[0];
    let dist_plane = &distorted.planes[0];
    let stride = ref_plane.stride;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let start_x = bx * block_size;
            let start_y = by * block_size;
            let end_x = (start_x + block_size).min(width);
            let end_y = (start_y + block_size).min(height);

            let mut mse_sum = 0.0;
            let mut pixel_count = 0;

            for y in start_y..end_y {
                for x in start_x..end_x {
                    let idx = y * stride + x;
                    if idx < ref_plane.data.len() && idx < dist_plane.data.len() {
                        let diff = f64::from(ref_plane.data[idx]) - f64::from(dist_plane.data[idx]);
                        mse_sum += diff * diff;
                        pixel_count += 1;
                    }
                }
            }

            let mse = if pixel_count > 0 {
                mse_sum / pixel_count as f64
            } else {
                0.0
            };

            psnr_map[by][bx] = if mse < 1e-10 {
                100.0
            } else {
                10.0 * (max_val_sq / mse).log10()
            };
        }
    }

    Ok(psnr_map)
}

/// Calculate PSNR statistics over a sequence of frames.
///
/// Returns min, max, mean, and standard deviation of PSNR values.
///
/// # Errors
///
/// Returns an error if frame sequences are incompatible.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::psnr::calculate_psnr_statistics;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut ref_frames = vec![];
/// let mut dist_frames = vec![];
/// for _ in 0..30 {
///     let mut ref_frame = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
///     ref_frame.allocate();
///     ref_frames.push(ref_frame);
///
///     let mut dist_frame = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
///     dist_frame.allocate();
///     dist_frames.push(dist_frame);
/// }
///
/// let stats = calculate_psnr_statistics(&ref_frames, &dist_frames).expect("psnr statistics should succeed");
/// println!("Mean PSNR: {:.2} dB", stats.mean);
/// ```
pub fn calculate_psnr_statistics(
    reference_frames: &[VideoFrame],
    distorted_frames: &[VideoFrame],
) -> CvResult<PsnrStatistics> {
    if reference_frames.len() != distorted_frames.len() {
        return Err(CvError::invalid_parameter(
            "frame_count",
            format!("{} vs {}", reference_frames.len(), distorted_frames.len()),
        ));
    }

    if reference_frames.is_empty() {
        return Err(CvError::insufficient_data(1, 0));
    }

    let mut psnr_values = Vec::with_capacity(reference_frames.len());

    for (ref_frame, dist_frame) in reference_frames.iter().zip(distorted_frames.iter()) {
        let psnr = calculate_psnr(ref_frame, dist_frame)?;
        psnr_values.push(psnr);
    }

    let min = psnr_values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = psnr_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean = psnr_values.iter().sum::<f64>() / psnr_values.len() as f64;

    let variance = psnr_values
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f64>()
        / psnr_values.len() as f64;
    let std_dev = variance.sqrt();

    // Calculate percentiles
    let mut sorted = psnr_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile_1 = sorted[sorted.len() / 100];
    let percentile_5 = sorted[sorted.len() * 5 / 100];
    let percentile_95 = sorted[sorted.len() * 95 / 100];
    let percentile_99 = sorted[sorted.len() * 99 / 100];
    let median = sorted[sorted.len() / 2];

    Ok(PsnrStatistics {
        min,
        max,
        mean,
        std_dev,
        median,
        percentile_1,
        percentile_5,
        percentile_95,
        percentile_99,
        per_frame: psnr_values,
    })
}

/// PSNR statistics for a video sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct PsnrStatistics {
    /// Minimum PSNR across all frames.
    pub min: f64,
    /// Maximum PSNR across all frames.
    pub max: f64,
    /// Mean PSNR.
    pub mean: f64,
    /// Standard deviation of PSNR.
    pub std_dev: f64,
    /// Median PSNR.
    pub median: f64,
    /// 1st percentile PSNR.
    pub percentile_1: f64,
    /// 5th percentile PSNR.
    pub percentile_5: f64,
    /// 95th percentile PSNR.
    pub percentile_95: f64,
    /// 99th percentile PSNR.
    pub percentile_99: f64,
    /// Per-frame PSNR values.
    pub per_frame: Vec<f64>,
}

impl PsnrStatistics {
    /// Get quality category based on mean PSNR.
    #[must_use]
    pub fn quality_category(&self) -> PsnrQuality {
        PsnrQuality::from_psnr(self.mean)
    }

    /// Check if quality is consistent (low std dev).
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.std_dev < 3.0
    }

    /// Get coefficient of variation (std_dev / mean).
    #[must_use]
    pub fn coefficient_of_variation(&self) -> f64 {
        self.std_dev / self.mean
    }
}
