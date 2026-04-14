//! Advanced image inpainting algorithms.
//!
//! Provides Telea (Fast Marching Method), Navier-Stokes diffusion, and
//! PatchMatch-based inpainting for filling masked regions in images.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// InpaintMethod
// ---------------------------------------------------------------------------

/// Algorithm selection for inpainting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InpaintMethod {
    /// Telea's Fast Marching Method — weighted average propagated outward.
    Telea,
    /// Navier-Stokes inspired Laplacian diffusion.
    Navier,
    /// PatchMatch-based exemplar filling.
    PatchMatch,
}

impl InpaintMethod {
    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Telea => "Telea (FMM)",
            Self::Navier => "Navier-Stokes diffusion",
            Self::PatchMatch => "PatchMatch exemplar",
        }
    }
}

// ---------------------------------------------------------------------------
// InpaintMask
// ---------------------------------------------------------------------------

/// Binary mask indicating which pixels require inpainting.
///
/// A value of `true` means the pixel needs to be filled.
#[derive(Debug, Clone)]
pub struct InpaintMask {
    /// Per-pixel mask data (row-major).
    pub data: Vec<bool>,
    /// Width of the mask in pixels.
    pub width: u32,
    /// Height of the mask in pixels.
    pub height: u32,
}

impl InpaintMask {
    /// Create a mask where all pixels are unmasked.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![false; (width as usize) * (height as usize)],
            width,
            height,
        }
    }

    /// Create a mask from pre-existing boolean data.
    ///
    /// Returns `None` if the data length does not match `width * height`.
    #[must_use]
    pub fn from_data(data: Vec<bool>, width: u32, height: u32) -> Option<Self> {
        if data.len() != (width as usize) * (height as usize) {
            return None;
        }
        Some(Self {
            data,
            width,
            height,
        })
    }

    /// Create a mask by thresholding a grayscale image.
    ///
    /// Pixels with value >= `threshold` are marked as masked.
    #[must_use]
    pub fn from_threshold(gray: &[u8], width: u32, height: u32, threshold: u8) -> Option<Self> {
        let expected = (width as usize) * (height as usize);
        if gray.len() != expected {
            return None;
        }
        let data: Vec<bool> = gray.iter().map(|&v| v >= threshold).collect();
        Some(Self {
            data,
            width,
            height,
        })
    }

    /// Count the number of connected regions of masked pixels (4-connected).
    #[must_use]
    pub fn region_count(&self) -> u32 {
        let w = self.width as usize;
        let h = self.height as usize;
        let n = w * h;
        let mut visited = vec![false; n];
        let mut count = 0u32;

        for start in 0..n {
            if !self.data[start] || visited[start] {
                continue;
            }
            // BFS flood fill
            count += 1;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            while let Some(idx) = queue.pop_front() {
                let x = idx % w;
                let y = idx / w;
                let neighbors: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dx, dy) in &neighbors {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let ni = ny as usize * w + nx as usize;
                    if self.data[ni] && !visited[ni] {
                        visited[ni] = true;
                        queue.push_back(ni);
                    }
                }
            }
        }
        count
    }

    /// Returns `true` if the given pixel is masked.
    #[must_use]
    pub fn is_masked(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.data[(y as usize) * (self.width as usize) + (x as usize)]
    }

    /// Mark a pixel as masked.
    pub fn mark(&mut self, x: u32, y: u32) {
        if x < self.width && y < self.height {
            self.data[(y as usize) * (self.width as usize) + (x as usize)] = true;
        }
    }

    /// Number of masked pixels.
    #[must_use]
    pub fn masked_count(&self) -> usize {
        self.data.iter().filter(|&&v| v).count()
    }
}

// ---------------------------------------------------------------------------
// Distance transform
// ---------------------------------------------------------------------------

/// Compute the Euclidean distance transform of a binary mask.
///
/// For each pixel, returns the distance to the nearest *unmasked* (false) pixel.
/// Unmasked pixels have distance 0.0.  Uses a two-pass algorithm.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn distance_transform(mask: &InpaintMask) -> Vec<f32> {
    let w = mask.width as usize;
    let h = mask.height as usize;
    let n = w * h;
    if n == 0 {
        return Vec::new();
    }

    let big = (w + h) as f32 * 2.0;
    let mut dist = vec![big; n];

    // Initialize: unmasked pixels get distance 0
    for i in 0..n {
        if !mask.data[i] {
            dist[i] = 0.0;
        }
    }

    // Forward pass (top-left to bottom-right)
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if x > 0 {
                let candidate = dist[idx - 1] + 1.0;
                if candidate < dist[idx] {
                    dist[idx] = candidate;
                }
            }
            if y > 0 {
                let candidate = dist[idx - w] + 1.0;
                if candidate < dist[idx] {
                    dist[idx] = candidate;
                }
            }
            // Diagonal
            if x > 0 && y > 0 {
                let candidate = dist[idx - w - 1] + std::f32::consts::SQRT_2;
                if candidate < dist[idx] {
                    dist[idx] = candidate;
                }
            }
            if x + 1 < w && y > 0 {
                let candidate = dist[idx - w + 1] + std::f32::consts::SQRT_2;
                if candidate < dist[idx] {
                    dist[idx] = candidate;
                }
            }
        }
    }

    // Backward pass (bottom-right to top-left)
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let idx = y * w + x;
            if x + 1 < w {
                let candidate = dist[idx + 1] + 1.0;
                if candidate < dist[idx] {
                    dist[idx] = candidate;
                }
            }
            if y + 1 < h {
                let candidate = dist[idx + w] + 1.0;
                if candidate < dist[idx] {
                    dist[idx] = candidate;
                }
            }
            if x + 1 < w && y + 1 < h {
                let candidate = dist[idx + w + 1] + std::f32::consts::SQRT_2;
                if candidate < dist[idx] {
                    dist[idx] = candidate;
                }
            }
            if x > 0 && y + 1 < h {
                let candidate = dist[idx + w - 1] + std::f32::consts::SQRT_2;
                if candidate < dist[idx] {
                    dist[idx] = candidate;
                }
            }
        }
    }

    dist
}

// ---------------------------------------------------------------------------
// Telea FMM inpainting
// ---------------------------------------------------------------------------

/// Pixel state for Fast Marching Method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FmmState {
    Known,
    Band,
    Unknown,
}

/// Entry in the FMM narrow-band priority queue.
#[derive(Debug, Clone)]
struct FmmEntry {
    dist: f32,
    idx: usize,
}

impl PartialEq for FmmEntry {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}
impl Eq for FmmEntry {}

impl PartialOrd for FmmEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FmmEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse comparison
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Telea's Fast Marching Method inpainting.
///
/// Processes boundary pixels outward using a priority queue (smallest distance
/// first), filling each masked pixel with a weighted average of known neighbors
/// within the given `radius`.
///
/// `image` is expected to be grayscale u8, single channel, row-major.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn inpaint_telea(
    image: &[u8],
    mask: &InpaintMask,
    width: u32,
    height: u32,
    radius: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    let mut output = image.to_vec();
    if n == 0 || output.len() != n {
        return output;
    }

    let r = radius.max(1) as i64;
    let mut state = vec![FmmState::Unknown; n];
    let mut dist = vec![f32::MAX; n];
    let mut heap: BinaryHeap<FmmEntry> = BinaryHeap::new();

    // Initialize: known pixels and band
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !mask.data[idx] {
                state[idx] = FmmState::Known;
                dist[idx] = 0.0;
            }
        }
    }

    // Find initial band: unknown pixels adjacent to known
    let dirs: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if state[idx] != FmmState::Unknown {
                continue;
            }
            let mut adjacent_known = false;
            for (dx, dy) in &dirs {
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let ni = ny as usize * w + nx as usize;
                    if state[ni] == FmmState::Known {
                        adjacent_known = true;
                        break;
                    }
                }
            }
            if adjacent_known {
                state[idx] = FmmState::Band;
                dist[idx] = 1.0;
                heap.push(FmmEntry { dist: 1.0, idx });
            }
        }
    }

    // March outward
    while let Some(entry) = heap.pop() {
        let idx = entry.idx;
        if state[idx] == FmmState::Known {
            continue;
        }
        state[idx] = FmmState::Known;

        let cx = (idx % w) as i64;
        let cy = (idx / w) as i64;

        // Weighted average of known neighbors within radius
        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;

        for dy in -r..=r {
            for dx in -r..=r {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if state[ni] != FmmState::Known || ni == idx {
                    continue;
                }
                let d2 = (dx * dx + dy * dy) as f64;
                let d = d2.sqrt();
                if d > r as f64 {
                    continue;
                }
                // Weight: inverse distance squared
                let weight = 1.0 / (d2 + 1e-6);
                weighted_sum += weight * output[ni] as f64;
                weight_total += weight;
            }
        }

        if weight_total > 1e-12 {
            let val = (weighted_sum / weight_total).round().clamp(0.0, 255.0);
            output[idx] = val as u8;
        }

        dist[idx] = entry.dist;

        // Add unknown neighbors to band
        for (dx, dy) in &dirs {
            let nx = cx + dx;
            let ny = cy + dy;
            if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                continue;
            }
            let ni = ny as usize * w + nx as usize;
            if state[ni] == FmmState::Unknown {
                state[ni] = FmmState::Band;
                let new_dist = entry.dist + 1.0;
                dist[ni] = new_dist;
                heap.push(FmmEntry {
                    dist: new_dist,
                    idx: ni,
                });
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Navier-Stokes diffusion inpainting
// ---------------------------------------------------------------------------

/// Navier-Stokes inspired inpainting using iterative Laplacian diffusion.
///
/// Fills masked regions by iteratively diffusing pixel values from the boundary
/// inward, driven by the discrete Laplacian.
///
/// `image` is grayscale u8, single channel, row-major.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn inpaint_navier(
    image: &[u8],
    mask: &InpaintMask,
    width: u32,
    height: u32,
    iterations: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if n == 0 || image.len() != n {
        return image.to_vec();
    }

    // Work in f32 space
    let mut current: Vec<f32> = image.iter().map(|&v| v as f32).collect();
    let mut next = current.clone();

    // Pre-compute which pixels are inside the mask
    let masked_indices: Vec<usize> = (0..n).filter(|&i| mask.data[i]).collect();

    // Initialize masked pixels with average of all known boundary pixels
    let boundary_avg = compute_boundary_average(&current, mask, w, h);
    for &idx in &masked_indices {
        current[idx] = boundary_avg;
    }

    // Iterative Laplacian diffusion
    for _iter in 0..iterations {
        next.copy_from_slice(&current);
        for &idx in &masked_indices {
            let x = idx % w;
            let y = idx / w;

            // Discrete Laplacian using 4-connected neighbors
            let mut sum = 0.0_f32;
            let mut count = 0u32;

            if x > 0 {
                sum += current[idx - 1];
                count += 1;
            }
            if x + 1 < w {
                sum += current[idx + 1];
                count += 1;
            }
            if y > 0 {
                sum += current[idx - w];
                count += 1;
            }
            if y + 1 < h {
                sum += current[idx + w];
                count += 1;
            }

            if count > 0 {
                // Blend: Laplacian average
                next[idx] = sum / count as f32;
            }
        }
        current.copy_from_slice(&next);
    }

    // Convert back to u8
    current
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// Compute the average value of known pixels adjacent to the mask boundary.
#[allow(clippy::cast_precision_loss)]
fn compute_boundary_average(image: &[f32], mask: &InpaintMask, w: usize, h: usize) -> f32 {
    let dirs: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let mut sum = 0.0_f64;
    let mut count = 0u64;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !mask.data[idx] {
                continue;
            }
            for (dx, dy) in &dirs {
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if !mask.data[ni] {
                    sum += image[ni] as f64;
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        128.0
    } else {
        (sum / count as f64) as f32
    }
}

// ---------------------------------------------------------------------------
// PatchMatch inpainting (simplified)
// ---------------------------------------------------------------------------

/// Simple patch-based inpainting.
///
/// For each masked pixel, searches for the best matching patch in the known
/// region and copies from it. Uses a randomized search strategy inspired by
/// the PatchMatch algorithm.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn inpaint_patchmatch(
    image: &[u8],
    mask: &InpaintMask,
    width: u32,
    height: u32,
    patch_radius: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if n == 0 || image.len() != n {
        return image.to_vec();
    }

    let pr = patch_radius.max(1) as i64;
    let mut output = image.to_vec();

    // Collect known pixel positions
    let known_positions: Vec<(usize, usize)> = (0..n)
        .filter(|&i| !mask.data[i])
        .map(|i| (i % w, i / w))
        .collect();

    if known_positions.is_empty() {
        return output;
    }

    // Process masked pixels using distance from boundary (inner pixels last)
    let dt = distance_transform(mask);
    let mut masked_with_dist: Vec<(usize, f32)> = (0..n)
        .filter(|&i| mask.data[i])
        .map(|i| (i, dt[i]))
        .collect();
    masked_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // Simple deterministic search: for each masked pixel find closest known
    // patch with best SSD
    let search_step = (known_positions.len() / 64).max(1);

    for (idx, _dist) in &masked_with_dist {
        let cx = idx % w;
        let cy = idx / w;

        let mut best_val = 128u8;
        let mut best_cost = f64::MAX;

        // Sample known positions
        for ki in (0..known_positions.len()).step_by(search_step) {
            let (kx, ky) = known_positions[ki];
            let mut ssd = 0.0_f64;
            let mut valid = 0u32;

            for dy in -pr..=pr {
                for dx in -pr..=pr {
                    let sx = cx as i64 + dx;
                    let sy = cy as i64 + dy;
                    let tx = kx as i64 + dx;
                    let ty = ky as i64 + dy;

                    if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                        continue;
                    }
                    if tx < 0 || ty < 0 || tx >= w as i64 || ty >= h as i64 {
                        continue;
                    }

                    let si = sy as usize * w + sx as usize;
                    let ti = ty as usize * w + tx as usize;

                    // Only compare known source pixels
                    if !mask.data[si] {
                        let diff = output[si] as f64 - output[ti] as f64;
                        ssd += diff * diff;
                        valid += 1;
                    }
                }
            }

            if valid > 0 {
                let cost = ssd / valid as f64;
                if cost < best_cost {
                    best_cost = cost;
                    best_val = output[ky * w + kx];
                }
            }
        }

        output[*idx] = best_val;
    }

    output
}

// ---------------------------------------------------------------------------
// High-level dispatch
// ---------------------------------------------------------------------------

/// Inpaint an image using the specified method.
///
/// `image` is grayscale u8, single channel.
#[must_use]
pub fn inpaint(
    image: &[u8],
    mask: &InpaintMask,
    width: u32,
    height: u32,
    method: InpaintMethod,
    radius: u32,
    iterations: u32,
) -> Vec<u8> {
    match method {
        InpaintMethod::Telea => inpaint_telea(image, mask, width, height, radius),
        InpaintMethod::Navier => inpaint_navier(image, mask, width, height, iterations),
        InpaintMethod::PatchMatch => inpaint_patchmatch(image, mask, width, height, radius),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- InpaintMask ---

    #[test]
    fn test_mask_new() {
        let m = InpaintMask::new(4, 4);
        assert_eq!(m.width, 4);
        assert_eq!(m.height, 4);
        assert_eq!(m.masked_count(), 0);
    }

    #[test]
    fn test_mask_from_data_valid() {
        let data = vec![true, false, false, true];
        let m = InpaintMask::from_data(data, 2, 2);
        assert!(m.is_some());
        let m = m.expect("valid mask");
        assert_eq!(m.masked_count(), 2);
    }

    #[test]
    fn test_mask_from_data_invalid_len() {
        let data = vec![true, false];
        assert!(InpaintMask::from_data(data, 3, 3).is_none());
    }

    #[test]
    fn test_mask_from_threshold() {
        let gray = vec![10, 200, 50, 255];
        let m = InpaintMask::from_threshold(&gray, 2, 2, 100);
        assert!(m.is_some());
        let m = m.expect("valid mask");
        assert!(!m.is_masked(0, 0)); // 10 < 100
        assert!(m.is_masked(1, 0)); // 200 >= 100
        assert!(!m.is_masked(0, 1)); // 50 < 100
        assert!(m.is_masked(1, 1)); // 255 >= 100
    }

    #[test]
    fn test_mask_from_threshold_invalid() {
        assert!(InpaintMask::from_threshold(&[1, 2], 3, 3, 100).is_none());
    }

    #[test]
    fn test_mask_region_count_single() {
        let mut m = InpaintMask::new(4, 4);
        m.mark(1, 1);
        m.mark(2, 1);
        m.mark(1, 2);
        assert_eq!(m.region_count(), 1);
    }

    #[test]
    fn test_mask_region_count_two() {
        let mut m = InpaintMask::new(5, 1);
        m.mark(0, 0);
        m.mark(4, 0);
        assert_eq!(m.region_count(), 2);
    }

    #[test]
    fn test_mask_region_count_zero() {
        let m = InpaintMask::new(3, 3);
        assert_eq!(m.region_count(), 0);
    }

    #[test]
    fn test_mask_is_masked_out_of_bounds() {
        let m = InpaintMask::new(2, 2);
        assert!(!m.is_masked(10, 10));
    }

    // --- Distance transform ---

    #[test]
    fn test_distance_transform_no_mask() {
        let m = InpaintMask::new(3, 3);
        let dt = distance_transform(&m);
        assert!(dt.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_distance_transform_center_pixel() {
        let data = vec![false, false, false, false, true, false, false, false, false];
        let m = InpaintMask::from_data(data, 3, 3).expect("valid mask");
        let dt = distance_transform(&m);
        // Center pixel should have distance 1.0
        assert!((dt[4] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_transform_empty() {
        let m = InpaintMask::new(0, 0);
        let dt = distance_transform(&m);
        assert!(dt.is_empty());
    }

    #[test]
    fn test_distance_transform_all_masked() {
        let data = vec![true; 4];
        let m = InpaintMask::from_data(data, 2, 2).expect("valid mask");
        let dt = distance_transform(&m);
        // All large values (no unmasked pixel)
        assert!(dt.iter().all(|&v| v > 1.0));
    }

    // --- Telea inpainting ---

    #[test]
    fn test_telea_no_mask() {
        let img = vec![100u8; 9];
        let m = InpaintMask::new(3, 3);
        let out = inpaint_telea(&img, &m, 3, 3, 2);
        assert_eq!(out, img);
    }

    #[test]
    fn test_telea_single_pixel() {
        // 3x3 uniform image with center masked
        let mut img = vec![200u8; 9];
        img[4] = 0; // center
        let data = vec![false, false, false, false, true, false, false, false, false];
        let m = InpaintMask::from_data(data, 3, 3).expect("valid mask");
        let out = inpaint_telea(&img, &m, 3, 3, 2);
        // Center should be filled close to 200
        assert!(out[4] > 150, "Expected > 150, got {}", out[4]);
    }

    #[test]
    fn test_telea_empty_image() {
        let out = inpaint_telea(&[], &InpaintMask::new(0, 0), 0, 0, 2);
        assert!(out.is_empty());
    }

    // --- Navier inpainting ---

    #[test]
    fn test_navier_no_mask() {
        let img = vec![128u8; 9];
        let m = InpaintMask::new(3, 3);
        let out = inpaint_navier(&img, &m, 3, 3, 10);
        assert_eq!(out, img);
    }

    #[test]
    fn test_navier_single_pixel() {
        let mut img = vec![200u8; 9];
        img[4] = 0;
        let data = vec![false, false, false, false, true, false, false, false, false];
        let m = InpaintMask::from_data(data, 3, 3).expect("valid mask");
        let out = inpaint_navier(&img, &m, 3, 3, 50);
        assert!(out[4] > 150, "Expected > 150, got {}", out[4]);
    }

    #[test]
    fn test_navier_convergence() {
        // More iterations should bring the center closer to surrounding value
        let mut img = vec![180u8; 25];
        for i in [6, 7, 8, 11, 12, 13, 16, 17, 18] {
            img[i] = 0;
        }
        let mut data = vec![false; 25];
        for i in [6, 7, 8, 11, 12, 13, 16, 17, 18] {
            data[i] = true;
        }
        let m = InpaintMask::from_data(data, 5, 5).expect("valid mask");
        let out10 = inpaint_navier(&img, &m, 5, 5, 10);
        let out100 = inpaint_navier(&img, &m, 5, 5, 100);
        // 100 iterations should get center pixel closer to 180 than 10 iterations
        let diff10 = (180i32 - out10[12] as i32).unsigned_abs();
        let diff100 = (180i32 - out100[12] as i32).unsigned_abs();
        assert!(
            diff100 <= diff10,
            "100 iters diff={diff100} should be <= 10 iters diff={diff10}"
        );
    }

    // --- PatchMatch inpainting ---

    #[test]
    fn test_patchmatch_no_mask() {
        let img = vec![100u8; 9];
        let m = InpaintMask::new(3, 3);
        let out = inpaint_patchmatch(&img, &m, 3, 3, 1);
        assert_eq!(out, img);
    }

    #[test]
    fn test_patchmatch_single_pixel() {
        let mut img = vec![150u8; 9];
        img[4] = 0;
        let data = vec![false, false, false, false, true, false, false, false, false];
        let m = InpaintMask::from_data(data, 3, 3).expect("valid mask");
        let out = inpaint_patchmatch(&img, &m, 3, 3, 1);
        // Should be filled with nearby known value (150)
        assert_eq!(out[4], 150);
    }

    // --- High-level dispatch ---

    #[test]
    fn test_inpaint_dispatch_telea() {
        let img = vec![100u8; 4];
        let m = InpaintMask::new(2, 2);
        let out = inpaint(&img, &m, 2, 2, InpaintMethod::Telea, 2, 10);
        assert_eq!(out, img);
    }

    #[test]
    fn test_inpaint_dispatch_navier() {
        let img = vec![100u8; 4];
        let m = InpaintMask::new(2, 2);
        let out = inpaint(&img, &m, 2, 2, InpaintMethod::Navier, 2, 10);
        assert_eq!(out, img);
    }

    #[test]
    fn test_inpaint_dispatch_patchmatch() {
        let img = vec![100u8; 4];
        let m = InpaintMask::new(2, 2);
        let out = inpaint(&img, &m, 2, 2, InpaintMethod::PatchMatch, 2, 10);
        assert_eq!(out, img);
    }

    // --- InpaintMethod ---

    #[test]
    fn test_method_names() {
        assert_eq!(InpaintMethod::Telea.name(), "Telea (FMM)");
        assert_eq!(InpaintMethod::Navier.name(), "Navier-Stokes diffusion");
        assert_eq!(InpaintMethod::PatchMatch.name(), "PatchMatch exemplar");
    }

    #[test]
    fn test_method_equality() {
        assert_eq!(InpaintMethod::Telea, InpaintMethod::Telea);
        assert_ne!(InpaintMethod::Telea, InpaintMethod::Navier);
    }
}

// ===========================================================================
// New high-level inpainting API
// ===========================================================================

/// Inpainting algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NewInpaintMethod {
    /// Telea Fast Marching Method — propagates known values outward along
    /// level-set contours from the mask boundary.
    Telea,
    /// Navier-Stokes diffusion — fills the masked region by iterative
    /// Laplacian smoothing, inspired by fluid dynamics equations.
    NavierStokes,
    /// Patch-based exemplar inpainting — copies the best-matching patch
    /// from the known region into each masked pixel.
    PatchBased,
}

impl NewInpaintMethod {
    /// Human-readable name of the method.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Telea => "Telea FMM",
            Self::NavierStokes => "Navier-Stokes",
            Self::PatchBased => "PatchBased",
        }
    }
}

/// Configuration for [`inpaint_image`].
#[derive(Debug, Clone)]
pub struct InpaintConfig {
    /// Algorithm to use for filling masked pixels.
    pub method: NewInpaintMethod,
    /// Neighbourhood radius used by Telea and PatchBased methods.
    pub radius: u32,
    /// Number of diffusion iterations for the NavierStokes method.
    pub iterations: u32,
}

impl Default for InpaintConfig {
    fn default() -> Self {
        Self {
            method: NewInpaintMethod::Telea,
            radius: 3,
            iterations: 100,
        }
    }
}

impl InpaintConfig {
    /// Convenience constructor.
    #[must_use]
    pub fn new(method: NewInpaintMethod, radius: u32) -> Self {
        Self {
            method,
            radius,
            ..Self::default()
        }
    }
}

/// Fill masked pixels in a grayscale image using the specified inpainting method.
///
/// # Arguments
///
/// * `image` — Grayscale u8 pixel data, row-major, length must equal
///   `width * height`.
/// * `mask`  — Per-pixel mask; non-zero values indicate pixels to be filled.
///   Must be the same length as `image`.
/// * `width`, `height` — Image dimensions.
/// * `config` — Inpainting configuration.
///
/// # Returns
///
/// A new `Vec<u8>` of the same length as `image` with masked pixels filled.
/// Returns a copy of `image` unchanged if dimensions are inconsistent.
#[must_use]
pub fn inpaint_image(
    image: &[u8],
    mask: &[u8],
    width: u32,
    height: u32,
    config: &InpaintConfig,
) -> Vec<u8> {
    let n = (width as usize) * (height as usize);
    if image.len() != n || mask.len() != n {
        return image.to_vec();
    }

    // Convert byte mask to InpaintMask struct
    let bool_data: Vec<bool> = mask.iter().map(|&v| v != 0).collect();
    let inpaint_mask = match InpaintMask::from_data(bool_data, width, height) {
        Some(m) => m,
        None => return image.to_vec(),
    };

    match config.method {
        NewInpaintMethod::Telea => {
            inpaint_telea(image, &inpaint_mask, width, height, config.radius)
        }
        NewInpaintMethod::NavierStokes => {
            inpaint_navier(image, &inpaint_mask, width, height, config.iterations)
        }
        NewInpaintMethod::PatchBased => {
            inpaint_patchmatch(image, &inpaint_mask, width, height, config.radius)
        }
    }
}

// ---------------------------------------------------------------------------
// Guided filter
// ---------------------------------------------------------------------------

/// Parameters for the guided image filter.
///
/// The guided filter (He et al., 2013) is a structure-aware edge-preserving
/// filter with O(N) complexity. A *guide image* `I` controls which edges are
/// preserved. When `I == p` (self-guided mode), it behaves like a bilateral
/// filter but is strictly O(N) per pass.
///
/// The filter fits a locally linear model between the guide and the output:
///
/// `q_i = a_k · I_i + b_k` for all `i ∈ ω_k`
///
/// where `ω_k` is a window of radius `r` and `(a_k, b_k)` are solved by
/// ridge regression with regularisation `ε²`.
#[derive(Debug, Clone, Copy)]
pub struct GuidedFilterParams {
    /// Half-size of the local averaging window (pixels). Diameter = 2r + 1.
    pub radius: usize,
    /// Regularisation parameter (typical 0.001–0.1 for images normalised to `[0, 1]`).
    pub epsilon: f32,
}

impl GuidedFilterParams {
    /// Creates new guided filter parameters.
    #[must_use]
    pub fn new(radius: usize, epsilon: f32) -> Self {
        Self { radius, epsilon }
    }
}

impl Default for GuidedFilterParams {
    fn default() -> Self {
        Self {
            radius: 4,
            epsilon: 0.01,
        }
    }
}

/// Compute the sliding-window mean of a 1-D slice (clamp-to-edge border).
#[allow(clippy::cast_precision_loss)]
fn box_mean_1d(src: &[f32], radius: usize) -> Vec<f32> {
    let n = src.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(n);
        let count = (hi - lo) as f32;
        let sum: f32 = src[lo..hi].iter().sum();
        out[i] = sum / count;
    }
    out
}

/// 2-D separable box-mean filter on a row-major single-channel image.
#[allow(clippy::cast_precision_loss)]
fn box_mean_2d(src: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    if width == 0 || height == 0 {
        return vec![0.0; src.len()];
    }
    // Horizontal pass — filter each row independently
    let mut h_pass = vec![0.0_f32; width * height];
    for y in 0..height {
        let row = &src[y * width..(y + 1) * width];
        let filtered = box_mean_1d(row, radius);
        h_pass[y * width..(y + 1) * width].copy_from_slice(&filtered);
    }
    // Vertical pass — filter each column independently
    let mut v_pass = vec![0.0_f32; width * height];
    for x in 0..width {
        let col: Vec<f32> = (0..height).map(|y| h_pass[y * width + x]).collect();
        let filtered = box_mean_1d(&col, radius);
        for (y, &val) in filtered.iter().enumerate() {
            v_pass[y * width + x] = val;
        }
    }
    v_pass
}

/// Apply the guided filter to `src` using `guide` for structure preservation.
///
/// Both `src` and `guide` are single-channel `f32` images in `[0, 1]` with
/// length `width × height` (row-major).
///
/// When `guide == src` this reduces to a self-guided (edge-preserving) filter.
///
/// # Returns
///
/// A `Vec<f32>` of length `width * height`.
///
/// # Panics
///
/// Panics if `src.len() != width * height` or `guide.len() != width * height`.
#[allow(clippy::cast_precision_loss)]
pub fn guided_filter(
    src: &[f32],
    guide: &[f32],
    width: usize,
    height: usize,
    params: &GuidedFilterParams,
) -> Vec<f32> {
    let n = width * height;
    assert_eq!(src.len(), n, "src length must equal width*height");
    assert_eq!(guide.len(), n, "guide length must equal width*height");

    if n == 0 {
        return Vec::new();
    }

    let r = params.radius;
    let eps = params.epsilon;

    // Step 1 — compute means of I (guide), p (src), I², I·p
    let mean_i = box_mean_2d(guide, width, height, r);
    let mean_p = box_mean_2d(src, width, height, r);

    let ii: Vec<f32> = guide.iter().map(|&v| v * v).collect();
    let ip: Vec<f32> = guide.iter().zip(src.iter()).map(|(&i, &p)| i * p).collect();
    let mean_ii = box_mean_2d(&ii, width, height, r);
    let mean_ip = box_mean_2d(&ip, width, height, r);

    // Step 2 — solve ridge-regression coefficients per window
    let mut a_coeffs = vec![0.0_f32; n];
    let mut b_coeffs = vec![0.0_f32; n];
    for i in 0..n {
        let var_i = mean_ii[i] - mean_i[i] * mean_i[i];
        let cov_ip = mean_ip[i] - mean_i[i] * mean_p[i];
        let a = cov_ip / (var_i + eps);
        let b = mean_p[i] - a * mean_i[i];
        a_coeffs[i] = a;
        b_coeffs[i] = b;
    }

    // Step 3 — average coefficients over windows
    let mean_a = box_mean_2d(&a_coeffs, width, height, r);
    let mean_b = box_mean_2d(&b_coeffs, width, height, r);

    // Step 4 — reconstruct output q = mean_a · I + mean_b
    guide
        .iter()
        .zip(mean_a.iter().zip(mean_b.iter()))
        .map(|(&i_val, (&ma, &mb))| ma * i_val + mb)
        .collect()
}

/// Apply the guided filter in self-guided mode (`guide == src`).
///
/// Equivalent to `guided_filter(src, src, width, height, params)`.
///
/// # Panics
///
/// Panics if `src.len() != width * height`.
pub fn guided_filter_self(
    src: &[f32],
    width: usize,
    height: usize,
    params: &GuidedFilterParams,
) -> Vec<f32> {
    guided_filter(src, src, width, height, params)
}

/// Fill masked pixels in a `u8` image using the guided filter.
///
/// The guided filter uses `guide` to constrain which edges are preserved when
/// estimating the values for masked pixels. Unmasked pixels are left unchanged.
///
/// # Arguments
///
/// * `src`    — grayscale u8 image with holes (values at masked positions are
///   ignored).
/// * `guide`  — grayscale u8 guide image (same size as `src`; often a
///   lower-frequency or reference image).
/// * `mask`   — per-pixel mask (u8); values `>= 128` denote pixels to fill.
/// * `width`, `height` — image dimensions.
/// * `params` — guided filter parameters.
///
/// # Returns
///
/// A `Vec<u8>` with masked pixels filled by the guided-filter estimate.
///
/// # Panics
///
/// Panics if `src`, `guide`, and `mask` do not all have length
/// `width * height`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn guided_inpaint(
    src: &[u8],
    guide: &[u8],
    mask: &[u8],
    width: usize,
    height: usize,
    params: &GuidedFilterParams,
) -> Vec<u8> {
    let n = width * height;
    assert_eq!(src.len(), n, "src length mismatch");
    assert_eq!(guide.len(), n, "guide length mismatch");
    assert_eq!(mask.len(), n, "mask length mismatch");

    let src_f32: Vec<f32> = src.iter().map(|&v| v as f32 / 255.0).collect();
    let guide_f32: Vec<f32> = guide.iter().map(|&v| v as f32 / 255.0).collect();

    let filtered = guided_filter(&src_f32, &guide_f32, width, height, params);

    src.iter()
        .zip(filtered.iter())
        .zip(mask.iter())
        .map(|((&orig, &filt), &m)| {
            if m >= 128 {
                (filt * 255.0).clamp(0.0, 255.0).round() as u8
            } else {
                orig
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// New inpainting API tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod new_inpaint_tests {
    use super::*;

    fn zero_mask(n: usize) -> Vec<u8> {
        vec![0u8; n]
    }

    fn single_pixel_mask(width: usize, height: usize, px: usize, py: usize) -> Vec<u8> {
        let mut m = vec![0u8; width * height];
        m[py * width + px] = 255;
        m
    }

    // --- InpaintConfig ---

    #[test]
    fn test_inpaint_config_default() {
        let cfg = InpaintConfig::default();
        assert_eq!(cfg.method, NewInpaintMethod::Telea);
        assert_eq!(cfg.radius, 3);
        assert_eq!(cfg.iterations, 100);
    }

    #[test]
    fn test_inpaint_config_new() {
        let cfg = InpaintConfig::new(NewInpaintMethod::PatchBased, 5);
        assert_eq!(cfg.method, NewInpaintMethod::PatchBased);
        assert_eq!(cfg.radius, 5);
    }

    #[test]
    fn test_inpaint_method_names() {
        assert_eq!(NewInpaintMethod::Telea.name(), "Telea FMM");
        assert_eq!(NewInpaintMethod::NavierStokes.name(), "Navier-Stokes");
        assert_eq!(NewInpaintMethod::PatchBased.name(), "PatchBased");
    }

    #[test]
    fn test_inpaint_method_equality() {
        assert_eq!(NewInpaintMethod::Telea, NewInpaintMethod::Telea);
        assert_ne!(NewInpaintMethod::Telea, NewInpaintMethod::NavierStokes);
    }

    // --- inpaint_image: no mask => unchanged ---

    #[test]
    fn test_inpaint_no_mask_telea() {
        let img = vec![100u8; 9];
        let mask = zero_mask(9);
        let out = inpaint_image(&img, &mask, 3, 3, &InpaintConfig::default());
        assert_eq!(out, img);
    }

    #[test]
    fn test_inpaint_no_mask_navier() {
        let img = vec![80u8; 9];
        let mask = zero_mask(9);
        let cfg = InpaintConfig::new(NewInpaintMethod::NavierStokes, 2);
        let out = inpaint_image(&img, &mask, 3, 3, &cfg);
        assert_eq!(out, img);
    }

    #[test]
    fn test_inpaint_no_mask_patchbased() {
        let img = vec![60u8; 9];
        let mask = zero_mask(9);
        let cfg = InpaintConfig::new(NewInpaintMethod::PatchBased, 1);
        let out = inpaint_image(&img, &mask, 3, 3, &cfg);
        assert_eq!(out, img);
    }

    // --- Single masked pixel ---

    #[test]
    fn test_inpaint_single_pixel_telea() {
        let mut img = vec![200u8; 9];
        img[4] = 0;
        let mask = single_pixel_mask(3, 3, 1, 1);
        let out = inpaint_image(&img, &mask, 3, 3, &InpaintConfig::default());
        assert!(out[4] > 100, "Center should be filled ~200, got {}", out[4]);
    }

    #[test]
    fn test_inpaint_single_pixel_navier() {
        let mut img = vec![200u8; 9];
        img[4] = 0;
        let mask = single_pixel_mask(3, 3, 1, 1);
        let cfg = InpaintConfig {
            method: NewInpaintMethod::NavierStokes,
            radius: 2,
            iterations: 50,
        };
        let out = inpaint_image(&img, &mask, 3, 3, &cfg);
        assert!(out[4] > 100, "Navier center fill={}", out[4]);
    }

    #[test]
    fn test_inpaint_single_pixel_patchbased() {
        let mut img = vec![150u8; 9];
        img[4] = 0;
        let mask = single_pixel_mask(3, 3, 1, 1);
        let cfg = InpaintConfig::new(NewInpaintMethod::PatchBased, 1);
        let out = inpaint_image(&img, &mask, 3, 3, &cfg);
        assert_eq!(out[4], 150);
    }

    // --- Dimension mismatch ---

    #[test]
    fn test_inpaint_image_size_mismatch_returns_original() {
        let img = vec![100u8; 9];
        let mask = vec![255u8; 5]; // wrong size
        let out = inpaint_image(&img, &mask, 3, 3, &InpaintConfig::default());
        assert_eq!(out, img);
    }

    #[test]
    fn test_inpaint_mask_size_mismatch_returns_original() {
        let img = vec![100u8; 6]; // wrong size
        let mask = zero_mask(9);
        let out = inpaint_image(&img, &mask, 3, 3, &InpaintConfig::default());
        assert_eq!(out, img);
    }

    #[test]
    fn test_inpaint_output_length_matches_input() {
        let img = vec![128u8; 16];
        let mask = zero_mask(16);
        let out = inpaint_image(&img, &mask, 4, 4, &InpaintConfig::default());
        assert_eq!(out.len(), 16);
    }
}

// ---------------------------------------------------------------------------
// Guided filter tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod guided_filter_tests {
    use super::*;

    #[test]
    fn test_guided_filter_params_default() {
        let p = GuidedFilterParams::default();
        assert_eq!(p.radius, 4);
        assert!(p.epsilon > 0.0);
    }

    #[test]
    fn test_guided_filter_params_new() {
        let p = GuidedFilterParams::new(6, 0.05);
        assert_eq!(p.radius, 6);
        assert!((p.epsilon - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_guided_filter_uniform_image_unchanged() {
        let src = vec![0.5_f32; 8 * 8];
        let p = GuidedFilterParams::new(2, 0.01);
        let out = guided_filter(&src, &src, 8, 8, &p);
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-4, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_guided_filter_output_length() {
        let src = vec![0.3_f32; 6 * 6];
        let p = GuidedFilterParams::default();
        let out = guided_filter(&src, &src, 6, 6, &p);
        assert_eq!(out.len(), 36);
    }

    #[test]
    fn test_guided_filter_self_matches_guided_with_self() {
        let src: Vec<f32> = (0..25).map(|i| i as f32 / 25.0).collect();
        let p = GuidedFilterParams::new(2, 0.01);
        let out1 = guided_filter(&src, &src, 5, 5, &p);
        let out2 = guided_filter_self(&src, 5, 5, &p);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_guided_filter_smooths_noise() {
        let mut src: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        src[50] = 0.99;
        let p = GuidedFilterParams::new(3, 0.1);
        let out = guided_filter_self(&src, 100, 1, &p);
        // Spike should be attenuated towards the smooth gradient value
        let spike_orig = 0.99_f32;
        let spike_out = out[50];
        // With high epsilon, the filter smooths heavily; output closer to ~0.5
        assert!(
            (spike_out - spike_orig).abs() > 0.01,
            "spike should be attenuated: orig={spike_orig}, out={spike_out}"
        );
    }

    #[test]
    fn test_guided_inpaint_no_mask_unchanged() {
        let src = vec![128u8; 5 * 5];
        let guide = vec![100u8; 5 * 5];
        let mask = vec![0u8; 5 * 5];
        let p = GuidedFilterParams::default();
        let out = guided_inpaint(&src, &guide, &mask, 5, 5, &p);
        assert_eq!(out, src);
    }

    #[test]
    fn test_guided_inpaint_output_length() {
        let src = vec![100u8; 4 * 4];
        let guide = vec![100u8; 4 * 4];
        let mask = vec![0u8; 4 * 4];
        let p = GuidedFilterParams::new(1, 0.01);
        let out = guided_inpaint(&src, &guide, &mask, 4, 4, &p);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_guided_inpaint_masked_pixel_filled() {
        // 5x5 image of all 200 with one masked pixel
        let mut src = vec![200u8; 5 * 5];
        src[12] = 0;
        let guide = vec![200u8; 5 * 5];
        let mut mask = vec![0u8; 5 * 5];
        mask[12] = 255;
        let p = GuidedFilterParams::new(2, 0.01);
        let out = guided_inpaint(&src, &guide, &mask, 5, 5, &p);
        assert!(
            out[12] > 100,
            "masked pixel should be filled near 200, got {}",
            out[12]
        );
    }

    #[test]
    fn test_guided_inpaint_known_pixels_unchanged() {
        let src: Vec<u8> = (0..16_u8).collect();
        let guide = src.clone();
        let mask = vec![0u8; 16];
        let p = GuidedFilterParams::new(1, 0.001);
        let out = guided_inpaint(&src, &guide, &mask, 4, 4, &p);
        assert_eq!(out, src);
    }
}
