//! Inpainting algorithms for delogo filter.
//!
//! Contains PDE-based inpainting (Navier-Stokes) and fast marching method.

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_arguments)]

use super::{DelogoFilter, OrderedFloat, Rectangle};

impl DelogoFilter {
    /// Apply Navier-Stokes PDE-based inpainting.
    pub(crate) fn apply_inpainting(
        &self,
        data: &mut [u8],
        region: &Rectangle,
        width: u32,
        height: u32,
    ) {
        // Create mask for the region
        let mut mask = vec![false; (width * height) as usize];
        for y in region.y..region.bottom().min(height) {
            for x in region.x..region.right().min(width) {
                mask[(y * width + x) as usize] = true;
            }
        }

        // Iterative inpainting using Navier-Stokes equations
        let mut working = data.to_vec();

        for _ in 0..self.config.iterations {
            let mut updated = false;

            for y in region.y..region.bottom().min(height) {
                for x in region.x..region.right().min(width) {
                    let idx = (y * width + x) as usize;

                    if mask[idx] {
                        // Compute gradient from boundary
                        let grad = compute_gradient_ns(&working, x, y, width, height, &mask);

                        if grad.0 != 0.0 || grad.1 != 0.0 {
                            // Propagate information from boundary
                            let new_val =
                                propagate_isophote(&working, x, y, width, height, &mask, grad);
                            working[idx] = new_val;
                            updated = true;
                        }
                    }
                }
            }

            if !updated {
                break;
            }
        }

        data.copy_from_slice(&working);
    }

    /// Apply fast marching method inpainting.
    pub(crate) fn apply_fast_marching(
        &self,
        data: &mut [u8],
        region: &Rectangle,
        width: u32,
        height: u32,
    ) {
        // Distance transform from boundary
        let mut distance = vec![f32::INFINITY; (width * height) as usize];
        let mut heap = std::collections::BinaryHeap::new();

        // Initialize boundary
        for y in region.y.saturating_sub(1)..=(region.bottom() + 1).min(height - 1) {
            for x in region.x.saturating_sub(1)..=(region.right() + 1).min(width - 1) {
                if !region.contains(x, y) {
                    let idx = (y * width + x) as usize;
                    distance[idx] = 0.0;
                    heap.push(OrderedFloat(-0.0, idx));
                }
            }
        }

        // Fast marching
        let mut working = data.to_vec();

        while let Some(OrderedFloat(neg_dist, idx)) = heap.pop() {
            let dist = -neg_dist;
            let y = (idx as u32 / width) as u32;
            let x = (idx as u32 % width) as u32;

            if distance[idx] < dist {
                continue;
            }

            // Update neighbors
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }

                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let nidx = (ny as u32 * width + nx as u32) as usize;
                        let step_dist = ((dx * dx + dy * dy) as f32).sqrt();
                        let new_dist = dist + step_dist;

                        if new_dist < distance[nidx] {
                            distance[nidx] = new_dist;

                            // Interpolate value
                            working[nidx] = interpolate_from_boundary(
                                data, nx as u32, ny as u32, width, height, region,
                            );

                            heap.push(OrderedFloat(-new_dist, nidx));
                        }
                    }
                }
            }
        }

        // Copy result
        for y in region.y..region.bottom().min(height) {
            for x in region.x..region.right().min(width) {
                let idx = (y * width + x) as usize;
                data[idx] = working[idx];
            }
        }
    }

    /// Apply exemplar-based inpainting using patch matching.
    pub(crate) fn apply_exemplar_based(
        &self,
        data: &mut [u8],
        region: &Rectangle,
        width: u32,
        height: u32,
    ) {
        let patch_size = self.config.patch_size as i32;
        let _half_patch = patch_size / 2;

        let mut working = data.to_vec();
        let mut mask = vec![false; (width * height) as usize];

        for y in region.y..region.bottom().min(height) {
            for x in region.x..region.right().min(width) {
                mask[(y * width + x) as usize] = true;
            }
        }

        // Iterative filling
        for _ in 0..self.config.iterations.min(10) {
            let mut updated = false;

            for y in region.y..region.bottom().min(height) {
                for x in region.x..region.right().min(width) {
                    let idx = (y * width + x) as usize;

                    if mask[idx] {
                        // Find best matching patch
                        if let Some(patch_val) =
                            find_best_patch(&working, x, y, width, height, &mask, patch_size)
                        {
                            working[idx] = patch_val;
                            updated = true;
                        }
                    }
                }
            }

            if !updated {
                break;
            }
        }

        data.copy_from_slice(&working);
    }
}

/// Compute gradient at a position (Navier-Stokes variant).
pub(crate) fn compute_gradient_ns(
    data: &[u8],
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    mask: &[bool],
) -> (f32, f32) {
    let mut gx = 0.0f32;
    let mut gy = 0.0f32;
    let mut count = 0;

    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }

            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                let nidx = (ny as u32 * width + nx as u32) as usize;

                if !mask.get(nidx).copied().unwrap_or(false) {
                    let val = data.get(nidx).copied().unwrap_or(128) as f32;
                    gx += val * dx as f32;
                    gy += val * dy as f32;
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        let norm = (gx * gx + gy * gy).sqrt().max(1.0);
        (gx / norm, gy / norm)
    } else {
        (0.0, 0.0)
    }
}

/// Propagate isophote information.
pub(crate) fn propagate_isophote(
    data: &[u8],
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    mask: &[bool],
    _grad: (f32, f32),
) -> u8 {
    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                let nidx = (ny as u32 * width + nx as u32) as usize;

                if !mask.get(nidx).copied().unwrap_or(false) {
                    let val = data.get(nidx).copied().unwrap_or(128) as f32;
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    let weight = 1.0 / (dist + 0.1);

                    sum += val * weight;
                    weight_sum += weight;
                }
            }
        }
    }

    if weight_sum > 0.0 {
        (sum / weight_sum).round().clamp(0.0, 255.0) as u8
    } else {
        data.get((y * width + x) as usize).copied().unwrap_or(128)
    }
}

/// Interpolate value from boundary pixels.
pub(crate) fn interpolate_from_boundary(
    data: &[u8],
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    region: &Rectangle,
) -> u8 {
    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                let nx = nx as u32;
                let ny = ny as u32;

                if !region.contains(nx, ny) {
                    let nidx = (ny * width + nx) as usize;
                    let val = data.get(nidx).copied().unwrap_or(128) as f32;
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    let weight = 1.0 / (dist + 0.1);

                    sum += val * weight;
                    weight_sum += weight;
                }
            }
        }
    }

    if weight_sum > 0.0 {
        (sum / weight_sum).round().clamp(0.0, 255.0) as u8
    } else {
        128
    }
}

/// Find the best matching patch for a position.
pub(crate) fn find_best_patch(
    data: &[u8],
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    mask: &[bool],
    patch_size: i32,
) -> Option<u8> {
    let half_patch = patch_size / 2;
    let search_range = 20;

    let mut best_diff = f32::INFINITY;
    let mut best_val = None;

    for sy in (y as i32 - search_range).max(half_patch)
        ..=(y as i32 + search_range).min(height as i32 - half_patch - 1)
    {
        for sx in (x as i32 - search_range).max(half_patch)
            ..=(x as i32 + search_range).min(width as i32 - half_patch - 1)
        {
            let sidx = (sy as u32 * width + sx as u32) as usize;

            if !mask.get(sidx).copied().unwrap_or(false) {
                // Compare patches
                let diff =
                    patch_difference(data, x, y, sx as u32, sy as u32, width, mask, half_patch);

                if diff < best_diff {
                    best_diff = diff;
                    best_val = data.get(sidx).copied();
                }
            }
        }
    }

    best_val
}

/// Compute difference between two patches.
pub(crate) fn patch_difference(
    data: &[u8],
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
    width: u32,
    mask: &[bool],
    radius: i32,
) -> f32 {
    let mut diff = 0.0f32;
    let mut count = 0;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px1 = x1 as i32 + dx;
            let py1 = y1 as i32 + dy;
            let px2 = x2 as i32 + dx;
            let py2 = y2 as i32 + dy;

            if px1 >= 0 && px2 >= 0 && py1 >= 0 && py2 >= 0 {
                let idx1 = (py1 as u32 * width + px1 as u32) as usize;
                let idx2 = (py2 as u32 * width + px2 as u32) as usize;

                // Only compare where both are known
                if !mask.get(idx1).copied().unwrap_or(false)
                    && !mask.get(idx2).copied().unwrap_or(false)
                {
                    let v1 = data.get(idx1).copied().unwrap_or(128) as f32;
                    let v2 = data.get(idx2).copied().unwrap_or(128) as f32;
                    diff += (v1 - v2).abs();
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        diff / count as f32
    } else {
        f32::INFINITY
    }
}
