//! Perceptual hashing algorithms for image fingerprinting.
//!
//! This module implements multiple perceptual hashing algorithms:
//!
//! - **pHash** (Perceptual Hash): DCT-based hash resistant to scaling and compression
//! - **aHash** (Average Hash): Simple average-based hash, fast but less robust
//! - **dHash** (Difference Hash): Gradient-based hash, good for detecting edits
//! - **wHash** (Wavelet Hash): Wavelet-transform based hash, very robust
//!
//! All hashes support Hamming distance comparison for similarity detection.

use crate::error::{CvError, CvResult};
use rayon::prelude::*;

/// Hash algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    /// Average hash - fast, simple, least robust.
    Average,
    /// Difference hash - gradient-based, good for edits.
    Difference,
    /// Perceptual hash - DCT-based, most robust.
    Perceptual,
    /// Wavelet hash - wavelet-based, very robust.
    Wavelet,
}

/// Computes perceptual hashes for multiple images in parallel.
///
/// # Arguments
///
/// * `images` - Slice of images (width, height, RGB data)
/// * `hash_size` - Hash size (8, 16, or 32)
///
/// # Errors
///
/// Returns an error if hashing fails.
pub fn compute_hashes_parallel(
    images: &[(u32, u32, Vec<u8>)],
    hash_size: usize,
) -> CvResult<Vec<u64>> {
    images
        .par_iter()
        .map(|(w, h, data)| compute_phash(data, *w, *h, hash_size))
        .collect()
}

/// Computes perceptual hashes for multiple images sequentially.
///
/// # Arguments
///
/// * `images` - Slice of images (width, height, RGB data)
/// * `hash_size` - Hash size (8, 16, or 32)
///
/// # Errors
///
/// Returns an error if hashing fails.
pub fn compute_hashes(images: &[(u32, u32, Vec<u8>)], hash_size: usize) -> CvResult<Vec<u64>> {
    images
        .iter()
        .map(|(w, h, data)| compute_phash(data, *w, *h, hash_size))
        .collect()
}

/// Computes perceptual hash (pHash) using DCT.
///
/// # Arguments
///
/// * `rgb_data` - RGB image data (row-major, 3 bytes per pixel)
/// * `width` - Image width
/// * `height` - Image height
/// * `hash_size` - Hash size (8, 16, or 32)
///
/// # Errors
///
/// Returns an error if image dimensions are invalid.
#[allow(clippy::similar_names)]
pub fn compute_phash(rgb_data: &[u8], width: u32, height: u32, hash_size: usize) -> CvResult<u64> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected_len = (width * height * 3) as usize;
    if rgb_data.len() != expected_len {
        return Err(CvError::insufficient_data(expected_len, rgb_data.len()));
    }

    // Convert to grayscale
    let gray = rgb_to_grayscale(rgb_data, width, height);

    // Resize to hash_size x hash_size
    let small_size = hash_size + 1; // +1 for DCT
    let resized = resize_bilinear(&gray, width, height, small_size as u32, small_size as u32);

    // Compute DCT
    let dct = compute_dct(&resized, small_size);

    // Extract low frequencies (top-left corner, excluding DC component)
    let mut values = Vec::with_capacity(hash_size * hash_size);
    for y in 0..hash_size {
        for x in 0..hash_size {
            if x == 0 && y == 0 {
                continue; // Skip DC component
            }
            values.push(dct[y * small_size + x]);
        }
    }

    // Compute median
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    // Generate hash
    let mut hash: u64 = 0;
    for (i, &val) in values.iter().enumerate() {
        if i >= 64 {
            break; // u64 has only 64 bits
        }
        if val > median {
            hash |= 1u64 << i;
        }
    }

    Ok(hash)
}

/// Computes average hash (aHash).
///
/// This is the simplest and fastest hash, but least robust to modifications.
///
/// # Arguments
///
/// * `rgb_data` - RGB image data
/// * `width` - Image width
/// * `height` - Image height
/// * `hash_size` - Hash size (typically 8)
///
/// # Errors
///
/// Returns an error if image dimensions are invalid.
pub fn compute_ahash(rgb_data: &[u8], width: u32, height: u32, hash_size: usize) -> CvResult<u64> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected_len = (width * height * 3) as usize;
    if rgb_data.len() != expected_len {
        return Err(CvError::insufficient_data(expected_len, rgb_data.len()));
    }

    // Convert to grayscale
    let gray = rgb_to_grayscale(rgb_data, width, height);

    // Resize to hash_size x hash_size
    let resized = resize_bilinear(&gray, width, height, hash_size as u32, hash_size as u32);

    // Compute average
    let sum: f32 = resized.iter().sum();
    let avg = sum / (hash_size * hash_size) as f32;

    // Generate hash
    let mut hash: u64 = 0;
    for (i, &val) in resized.iter().enumerate() {
        if i >= 64 {
            break;
        }
        if val > avg {
            hash |= 1u64 << i;
        }
    }

    Ok(hash)
}

/// Computes difference hash (dHash).
///
/// This hash captures horizontal gradients and is good for detecting edits.
///
/// # Arguments
///
/// * `rgb_data` - RGB image data
/// * `width` - Image width
/// * `height` - Image height
/// * `hash_size` - Hash size (typically 8)
///
/// # Errors
///
/// Returns an error if image dimensions are invalid.
pub fn compute_dhash(rgb_data: &[u8], width: u32, height: u32, hash_size: usize) -> CvResult<u64> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected_len = (width * height * 3) as usize;
    if rgb_data.len() != expected_len {
        return Err(CvError::insufficient_data(expected_len, rgb_data.len()));
    }

    // Convert to grayscale
    let gray = rgb_to_grayscale(rgb_data, width, height);

    // Resize to (hash_size + 1) x hash_size for horizontal gradient
    let resized = resize_bilinear(
        &gray,
        width,
        height,
        (hash_size + 1) as u32,
        hash_size as u32,
    );

    // Generate hash based on horizontal gradients
    let mut hash: u64 = 0;
    let mut bit = 0;
    for y in 0..hash_size {
        for x in 0..hash_size {
            if bit >= 64 {
                break;
            }
            let left = resized[y * (hash_size + 1) + x];
            let right = resized[y * (hash_size + 1) + x + 1];
            if left < right {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }

    Ok(hash)
}

/// Computes wavelet hash (wHash).
///
/// Uses discrete wavelet transform for robust hashing.
///
/// # Arguments
///
/// * `rgb_data` - RGB image data
/// * `width` - Image width
/// * `height` - Image height
/// * `hash_size` - Hash size (must be power of 2)
///
/// # Errors
///
/// Returns an error if image dimensions are invalid.
pub fn compute_whash(rgb_data: &[u8], width: u32, height: u32, hash_size: usize) -> CvResult<u64> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    if (hash_size & (hash_size - 1)) != 0 {
        return Err(CvError::invalid_parameter(
            "hash_size",
            format!("{hash_size} (must be power of 2)"),
        ));
    }

    let expected_len = (width * height * 3) as usize;
    if rgb_data.len() != expected_len {
        return Err(CvError::insufficient_data(expected_len, rgb_data.len()));
    }

    // Convert to grayscale
    let gray = rgb_to_grayscale(rgb_data, width, height);

    // Resize to hash_size x hash_size
    let resized = resize_bilinear(&gray, width, height, hash_size as u32, hash_size as u32);

    // Apply Haar wavelet transform
    let wavelet = haar_wavelet_2d(&resized, hash_size);

    // Use only the LL (low-low) subband for hashing
    let ll_size = hash_size / 2;
    let mut values = Vec::with_capacity(ll_size * ll_size);
    for y in 0..ll_size {
        for x in 0..ll_size {
            values.push(wavelet[y * hash_size + x]);
        }
    }

    // Compute median
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    // Generate hash
    let mut hash: u64 = 0;
    for (i, &val) in values.iter().enumerate() {
        if i >= 64 {
            break;
        }
        if val > median {
            hash |= 1u64 << i;
        }
    }

    Ok(hash)
}

/// Computes Hamming distance between two hashes.
///
/// Returns the number of differing bits (0-64).
#[must_use]
pub fn hamming_distance(hash1: u64, hash2: u64) -> u32 {
    (hash1 ^ hash2).count_ones()
}

/// Computes similarity score between two hashes.
///
/// Returns a value in [0.0, 1.0], where 1.0 means identical.
#[must_use]
pub fn hash_similarity(hash1: u64, hash2: u64) -> f64 {
    let distance = hamming_distance(hash1, hash2);
    1.0 - (f64::from(distance) / 64.0)
}

/// Converts RGB image to grayscale.
fn rgb_to_grayscale(rgb_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let mut gray = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let r = f32::from(rgb_data[idx]);
            let g = f32::from(rgb_data[idx + 1]);
            let b = f32::from(rgb_data[idx + 2]);

            // Standard RGB to grayscale conversion
            let gray_val = 0.299 * r + 0.587 * g + 0.114 * b;
            gray.push(gray_val);
        }
    }

    gray
}

/// Resizes image using bilinear interpolation.
#[allow(clippy::many_single_char_names)]
fn resize_bilinear(
    src: &[f32],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<f32> {
    let mut dst = vec![0.0; (dst_width * dst_height) as usize];

    let x_ratio = src_width as f32 / dst_width as f32;
    let y_ratio = src_height as f32 / dst_height as f32;

    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_x = x as f32 * x_ratio;
            let src_y = y as f32 * y_ratio;

            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);

            let dx = src_x - x0 as f32;
            let dy = src_y - y0 as f32;

            let p00 = src[(y0 * src_width + x0) as usize];
            let p10 = src[(y0 * src_width + x1) as usize];
            let p01 = src[(y1 * src_width + x0) as usize];
            let p11 = src[(y1 * src_width + x1) as usize];

            let val = p00 * (1.0 - dx) * (1.0 - dy)
                + p10 * dx * (1.0 - dy)
                + p01 * (1.0 - dx) * dy
                + p11 * dx * dy;

            dst[(y * dst_width + x) as usize] = val;
        }
    }

    dst
}

/// Computes 2D Discrete Cosine Transform.
fn compute_dct(data: &[f32], size: usize) -> Vec<f32> {
    let mut dct = vec![0.0; size * size];
    let n = size as f32;

    for v in 0..size {
        for u in 0..size {
            let mut sum = 0.0;

            for y in 0..size {
                for x in 0..size {
                    let pixel = data[y * size + x];
                    let cu = if u == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
                    let cv = if v == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };

                    let cos_u = ((2.0 * x as f32 + 1.0) * u as f32 * std::f32::consts::PI
                        / (2.0 * n))
                        .cos();
                    let cos_v = ((2.0 * y as f32 + 1.0) * v as f32 * std::f32::consts::PI
                        / (2.0 * n))
                        .cos();

                    sum += cu * cv * pixel * cos_u * cos_v;
                }
            }

            dct[v * size + u] = sum * 2.0 / n;
        }
    }

    dct
}

/// Applies 2D Haar wavelet transform.
fn haar_wavelet_2d(data: &[f32], size: usize) -> Vec<f32> {
    let mut result = data.to_vec();

    // Apply 1D Haar transform to rows
    let mut temp = vec![0.0; size];
    for y in 0..size {
        for x in 0..size {
            temp[x] = result[y * size + x];
        }
        haar_wavelet_1d(&mut temp);
        for x in 0..size {
            result[y * size + x] = temp[x];
        }
    }

    // Apply 1D Haar transform to columns
    for x in 0..size {
        for y in 0..size {
            temp[y] = result[y * size + x];
        }
        haar_wavelet_1d(&mut temp);
        for y in 0..size {
            result[y * size + x] = temp[y];
        }
    }

    result
}

/// Applies 1D Haar wavelet transform in-place.
fn haar_wavelet_1d(data: &mut [f32]) {
    let n = data.len();
    let mut temp = vec![0.0; n];
    let half = n / 2;

    let scale = 1.0 / 2.0_f32.sqrt();

    for i in 0..half {
        temp[i] = (data[2 * i] + data[2 * i + 1]) * scale;
        temp[half + i] = (data[2 * i] - data[2 * i + 1]) * scale;
    }

    data.copy_from_slice(&temp);
}

/// Batch computes hashes using specified algorithm.
///
/// # Errors
///
/// Returns an error if hashing fails.
pub fn compute_hashes_with_algorithm(
    images: &[(u32, u32, Vec<u8>)],
    hash_size: usize,
    algorithm: HashAlgorithm,
) -> CvResult<Vec<u64>> {
    images
        .iter()
        .map(|(w, h, data)| match algorithm {
            HashAlgorithm::Average => compute_ahash(data, *w, *h, hash_size),
            HashAlgorithm::Difference => compute_dhash(data, *w, *h, hash_size),
            HashAlgorithm::Perceptual => compute_phash(data, *w, *h, hash_size),
            HashAlgorithm::Wavelet => compute_whash(data, *w, *h, hash_size),
        })
        .collect()
}

/// Finds the best matching hash from a database.
///
/// Returns `(index, similarity)` of the best match, or `None` if no match above threshold.
#[must_use]
pub fn find_best_match(query: u64, database: &[u64], threshold: f64) -> Option<(usize, f64)> {
    let mut best_idx = 0;
    let mut best_similarity = 0.0;

    for (idx, &db_hash) in database.iter().enumerate() {
        let similarity = hash_similarity(query, db_hash);
        if similarity > best_similarity {
            best_similarity = similarity;
            best_idx = idx;
        }
    }

    if best_similarity >= threshold {
        Some((best_idx, best_similarity))
    } else {
        None
    }
}

/// Finds all matches above threshold.
#[must_use]
pub fn find_all_matches(query: u64, database: &[u64], threshold: f64) -> Vec<(usize, f64)> {
    database
        .iter()
        .enumerate()
        .filter_map(|(idx, &db_hash)| {
            let similarity = hash_similarity(query, db_hash);
            if similarity >= threshold {
                Some((idx, similarity))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let gray = ((x + y) % 256) as u8;
                data.push(gray);
                data.push(gray);
                data.push(gray);
            }
        }
        data
    }

    #[test]
    fn test_phash() {
        let img = create_test_image(64, 64);
        let hash = compute_phash(&img, 64, 64, 8).expect("compute_phash should succeed");
        assert!(hash > 0);
    }

    #[test]
    fn test_ahash() {
        let img = create_test_image(64, 64);
        let hash = compute_ahash(&img, 64, 64, 8).expect("compute_ahash should succeed");
        assert!(hash > 0);
    }

    #[test]
    fn test_dhash() {
        let img = create_test_image(64, 64);
        let hash = compute_dhash(&img, 64, 64, 8).expect("compute_dhash should succeed");
        assert!(hash > 0);
    }

    #[test]
    fn test_whash() {
        let img = create_test_image(64, 64);
        let hash = compute_whash(&img, 64, 64, 8).expect("compute_whash should succeed");
        assert!(hash > 0);
    }

    #[test]
    fn test_identical_images() {
        let img = create_test_image(64, 64);
        let hash1 = compute_phash(&img, 64, 64, 8).expect("compute_phash should succeed");
        let hash2 = compute_phash(&img, 64, 64, 8).expect("compute_phash should succeed");

        assert_eq!(hash1, hash2);
        assert_eq!(hamming_distance(hash1, hash2), 0);
        assert_eq!(hash_similarity(hash1, hash2), 1.0);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(0, 1), 1);
        assert_eq!(hamming_distance(0xFF, 0), 8);
        assert_eq!(hamming_distance(u64::MAX, 0), 64);
    }

    #[test]
    fn test_hash_similarity() {
        assert_eq!(hash_similarity(0, 0), 1.0);
        assert_eq!(hash_similarity(u64::MAX, 0), 0.0);
        assert!((hash_similarity(0, 1) - (63.0 / 64.0)).abs() < 0.01);
    }

    #[test]
    fn test_invalid_dimensions() {
        let img = create_test_image(0, 0);
        assert!(compute_phash(&img, 0, 0, 8).is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let img = vec![0u8; 100];
        assert!(compute_phash(&img, 64, 64, 8).is_err());
    }

    #[test]
    fn test_rgb_to_grayscale() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let gray = rgb_to_grayscale(&rgb, 3, 1);
        assert_eq!(gray.len(), 3);
        assert!(gray[0] > 0.0); // Red
        assert!(gray[1] > 0.0); // Green
        assert!(gray[2] > 0.0); // Blue
    }

    #[test]
    fn test_resize_bilinear() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let dst = resize_bilinear(&src, 2, 2, 1, 1);
        assert_eq!(dst.len(), 1);
        assert!(dst[0] > 0.0);
    }

    #[test]
    fn test_find_best_match() {
        let database = vec![
            0x0000000000000000,
            0x0000000000000001,
            0x000000000000000F,
            0x00000000000000FF,
        ];

        let result = find_best_match(0, &database, 0.95);
        assert!(result.is_some());
        let (idx, sim) = result.expect("operation should succeed");
        assert_eq!(idx, 0);
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_find_all_matches() {
        let database = vec![0, 1, 2, 3];
        let matches = find_all_matches(0, &database, 0.95);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_compute_hashes() {
        let images = vec![
            (64, 64, create_test_image(64, 64)),
            (64, 64, create_test_image(64, 64)),
        ];

        let hashes = compute_hashes(&images, 8).expect("compute_hashes should succeed");
        assert_eq!(hashes.len(), 2);
    }

    #[test]
    fn test_compute_hashes_parallel() {
        let images = vec![
            (64, 64, create_test_image(64, 64)),
            (64, 64, create_test_image(64, 64)),
            (64, 64, create_test_image(64, 64)),
        ];

        let hashes =
            compute_hashes_parallel(&images, 8).expect("compute_hashes_parallel should succeed");
        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn test_haar_wavelet_1d() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        haar_wavelet_1d(&mut data);
        assert_eq!(data.len(), 4);
    }

    #[test]
    fn test_dct() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let dct = compute_dct(&data, 2);
        assert_eq!(dct.len(), 4);
    }
}
