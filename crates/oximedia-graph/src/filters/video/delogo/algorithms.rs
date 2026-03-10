//! Spatial and temporal removal algorithms for delogo filter.
//!
//! Contains blur, edge-aware interpolation, temporal interpolation,
//! feathering, and logo detection methods.

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_arguments)]

use super::{create_gaussian_kernel, DelogoFilter, LogoDetection, Rectangle};
use oximedia_codec::Plane;

impl DelogoFilter {
    /// Apply Gaussian blur to the logo region.
    pub(crate) fn apply_blur(&self, data: &mut [u8], region: &Rectangle, width: u32, height: u32) {
        let radius = (region.width.min(region.height) / 8).clamp(3, 15);
        let sigma = radius as f32 / 2.0;

        // Create Gaussian kernel
        let kernel = create_gaussian_kernel(radius as usize, sigma);

        // Apply blur to region
        for y in region.y..region.bottom().min(height) {
            for x in region.x..region.right().min(width) {
                let blurred = apply_kernel(data, x, y, width, height, &kernel, radius as i32);
                let idx = (y * width + x) as usize;
                data[idx] = blurred;
            }
        }
    }

    /// Apply texture synthesis (delegates to exemplar-based).
    pub(crate) fn apply_texture_synthesis(
        &self,
        data: &mut [u8],
        region: &Rectangle,
        width: u32,
        height: u32,
    ) {
        // Use exemplar-based approach with larger search region
        self.apply_exemplar_based(data, region, width, height);
    }

    /// Apply edge-aware interpolation.
    pub(crate) fn apply_edge_aware(
        &self,
        data: &mut [u8],
        region: &Rectangle,
        width: u32,
        height: u32,
    ) {
        // Detect edges around the region
        let edges = detect_edges(data, width, height);

        let mut working = data.to_vec();

        for y in region.y..region.bottom().min(height) {
            for x in region.x..region.right().min(width) {
                let idx = (y * width + x) as usize;

                // Interpolate based on edge direction
                let edge_strength = edges[idx];
                let val = if edge_strength > 50.0 {
                    // Strong edge: interpolate along edge
                    interpolate_along_edge(data, x, y, width, height)
                } else {
                    // Weak edge: simple interpolation
                    super::inpainting::interpolate_from_boundary(data, x, y, width, height, region)
                };

                working[idx] = val;
            }
        }

        data.copy_from_slice(&working);
    }

    /// Apply temporal interpolation using neighboring frames.
    pub(crate) fn apply_temporal_interpolation(
        &self,
        data: &mut [u8],
        region: &Rectangle,
        width: u32,
        height: u32,
    ) {
        if self.frame_buffer.is_empty() {
            // Fall back to spatial inpainting
            self.apply_inpainting(data, region, width, height);
            return;
        }

        // Average corresponding pixels from temporal neighbors
        for y in region.y..region.bottom().min(height) {
            for x in region.x..region.right().min(width) {
                let idx = (y * width + x) as usize;

                let mut sum = 0.0f32;
                let mut count = 0;

                for frame in &self.frame_buffer {
                    if let Some(plane) = frame.planes.first() {
                        let fidx = (y * width + x) as usize;
                        if let Some(&val) = plane.data.get(fidx) {
                            sum += val as f32;
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    data[idx] = (sum / count as f32).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    /// Apply feathering/blending at region edges.
    pub(crate) fn apply_feathering(
        &self,
        data: &mut [u8],
        original: &Plane,
        region: &Rectangle,
        width: u32,
        height: u32,
    ) {
        let feather = self.config.feather as i32;
        let expanded = region.expand(self.config.feather);

        for y in expanded.y..expanded.bottom().min(height) {
            for x in expanded.x..expanded.right().min(width) {
                let idx = (y * width + x) as usize;

                // Compute distance to region boundary
                let dist = distance_to_boundary(x, y, region);

                if dist < feather as f32 {
                    // Blend between original and processed
                    let alpha = (dist / feather as f32).clamp(0.0, 1.0);
                    let alpha = alpha * self.config.strength;

                    let original_val = original.data.get(idx).copied().unwrap_or(128) as f32;
                    let processed_val = data.get(idx).copied().unwrap_or(128) as f32;

                    let blended = original_val * (1.0 - alpha) + processed_val * alpha;
                    data[idx] = blended.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    /// Detect logos in the frame.
    pub(crate) fn detect_logos(
        &mut self,
        frame: &oximedia_codec::VideoFrame,
        detection: &LogoDetection,
    ) -> Option<Vec<Rectangle>> {
        match detection {
            LogoDetection::Manual(region) => Some(vec![*region]),
            LogoDetection::Template {
                template,
                width: t_width,
                height: t_height,
                threshold,
            } => {
                if let Some(plane) = frame.planes.first() {
                    template_match(
                        &plane.data,
                        frame.width,
                        frame.height,
                        template,
                        *t_width,
                        *t_height,
                        *threshold,
                    )
                } else {
                    None
                }
            }
            LogoDetection::Automatic {
                search_region,
                sensitivity,
            } => {
                if let Some(plane) = frame.planes.first() {
                    auto_detect(
                        &plane.data,
                        frame.width,
                        frame.height,
                        search_region,
                        *sensitivity,
                    )
                } else {
                    None
                }
            }
        }
    }
}

/// Apply a convolution kernel at a specific position.
pub(crate) fn apply_kernel(
    data: &[u8],
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    kernel: &[f32],
    radius: i32,
) -> u8 {
    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    let ksize = (radius * 2 + 1) as usize;

    for ky in 0..ksize {
        let py = y as i32 + ky as i32 - radius;
        if py < 0 || py >= height as i32 {
            continue;
        }

        for kx in 0..ksize {
            let px = x as i32 + kx as i32 - radius;
            if px < 0 || px >= width as i32 {
                continue;
            }

            let idx = (py as u32 * width + px as u32) as usize;
            let weight = kernel[ky * ksize + kx];
            sum += data.get(idx).copied().unwrap_or(128) as f32 * weight;
            weight_sum += weight;
        }
    }

    if weight_sum > 0.0 {
        (sum / weight_sum).round().clamp(0.0, 255.0) as u8
    } else {
        128
    }
}

/// Detect edges in the image using Sobel operator.
pub(crate) fn detect_edges(data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let mut edges = vec![0.0f32; (width * height) as usize];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;

            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx) as u32;
                    let ny = (y as i32 + dy) as u32;
                    let nidx = (ny * width + nx) as usize;
                    let val = data.get(nidx).copied().unwrap_or(128) as f32;

                    // Sobel kernels
                    let sx = dx as f32;
                    let sy = dy as f32;

                    gx += val * sx;
                    gy += val * sy;
                }
            }

            let magnitude = (gx * gx + gy * gy).sqrt();
            edges[(y * width + x) as usize] = magnitude;
        }
    }

    edges
}

/// Interpolate along edge direction.
pub(crate) fn interpolate_along_edge(data: &[u8], x: u32, y: u32, width: u32, height: u32) -> u8 {
    // Simplified: average of perpendicular neighbors
    let mut sum = 0.0f32;
    let mut count = 0;

    for offset in [-2i32, -1, 1, 2] {
        for (dx, dy) in [(offset, 0), (0, offset)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                let nidx = (ny as u32 * width + nx as u32) as usize;
                sum += data.get(nidx).copied().unwrap_or(128) as f32;
                count += 1;
            }
        }
    }

    if count > 0 {
        (sum / count as f32).round().clamp(0.0, 255.0) as u8
    } else {
        128
    }
}

/// Compute distance from a point to the region boundary.
pub(crate) fn distance_to_boundary(x: u32, y: u32, region: &Rectangle) -> f32 {
    if !region.contains(x, y) {
        // Outside region
        let dx = if x < region.x {
            region.x - x
        } else if x >= region.right() {
            x - region.right() + 1
        } else {
            0
        };

        let dy = if y < region.y {
            region.y - y
        } else if y >= region.bottom() {
            y - region.bottom() + 1
        } else {
            0
        };

        ((dx * dx + dy * dy) as f32).sqrt()
    } else {
        // Inside region - distance to nearest edge
        let dx = (x - region.x).min(region.right() - x - 1);
        let dy = (y - region.y).min(region.bottom() - y - 1);
        dx.min(dy) as f32
    }
}

/// Template matching for logo detection.
fn template_match(
    data: &[u8],
    width: u32,
    height: u32,
    template: &[u8],
    t_width: u32,
    t_height: u32,
    threshold: f32,
) -> Option<Vec<Rectangle>> {
    let mut matches = Vec::new();

    for y in 0..=(height.saturating_sub(t_height)) {
        for x in 0..=(width.saturating_sub(t_width)) {
            let score = compute_template_score(data, x, y, width, template, t_width, t_height);

            if score >= threshold {
                matches.push(Rectangle::new(x, y, t_width, t_height));
            }
        }
    }

    if matches.is_empty() {
        None
    } else {
        Some(matches)
    }
}

/// Compute normalized cross-correlation score for template matching.
fn compute_template_score(
    data: &[u8],
    x: u32,
    y: u32,
    width: u32,
    template: &[u8],
    t_width: u32,
    t_height: u32,
) -> f32 {
    let mut sum = 0.0f32;
    let mut sq_sum = 0.0f32;
    let mut template_sum = 0.0f32;
    let mut template_sq_sum = 0.0f32;
    let mut cross_sum = 0.0f32;
    let mut count = 0;

    for ty in 0..t_height {
        for tx in 0..t_width {
            let px = x + tx;
            let py = y + ty;
            let idx = (py * width + px) as usize;
            let tidx = (ty * t_width + tx) as usize;

            let val = data.get(idx).copied().unwrap_or(0) as f32;
            let tval = template.get(tidx).copied().unwrap_or(0) as f32;

            sum += val;
            sq_sum += val * val;
            template_sum += tval;
            template_sq_sum += tval * tval;
            cross_sum += val * tval;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let n = count as f32;
    let numerator = cross_sum - (sum * template_sum / n);
    let denominator =
        ((sq_sum - sum * sum / n) * (template_sq_sum - template_sum * template_sum / n)).sqrt();

    if denominator > 0.0 {
        (numerator / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Automatic logo detection.
fn auto_detect(
    data: &[u8],
    width: u32,
    height: u32,
    search_region: &Rectangle,
    sensitivity: f32,
) -> Option<Vec<Rectangle>> {
    // Detect high-contrast regions that might be logos
    let edges = detect_edges(data, width, height);
    let threshold = 100.0 * sensitivity;

    let mut regions = Vec::new();

    // Simple connected component analysis
    let mut visited = vec![false; (width * height) as usize];

    for y in search_region.y..search_region.bottom().min(height) {
        for x in search_region.x..search_region.right().min(width) {
            let idx = (y * width + x) as usize;

            if !visited[idx] && edges.get(idx).copied().unwrap_or(0.0) > threshold {
                if let Some(region) =
                    extract_component(&edges, &mut visited, x, y, width, height, threshold)
                {
                    regions.push(region);
                }
            }
        }
    }

    if regions.is_empty() {
        None
    } else {
        Some(regions)
    }
}

/// Extract connected component.
fn extract_component(
    edges: &[f32],
    visited: &mut [bool],
    start_x: u32,
    start_y: u32,
    width: u32,
    height: u32,
    threshold: f32,
) -> Option<Rectangle> {
    let mut min_x = start_x;
    let mut max_x = start_x;
    let mut min_y = start_y;
    let mut max_y = start_y;

    let mut stack = vec![(start_x, start_y)];
    visited[(start_y * width + start_x) as usize] = true;

    while let Some((x, y)) = stack.pop() {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nidx = (ny as u32 * width + nx as u32) as usize;

                    if !visited[nidx] && edges.get(nidx).copied().unwrap_or(0.0) > threshold {
                        visited[nidx] = true;
                        stack.push((nx as u32, ny as u32));
                    }
                }
            }
        }
    }

    let w = max_x - min_x + 1;
    let h = max_y - min_y + 1;

    // Filter out very small or very large regions
    if w >= 10 && h >= 10 && w < width / 2 && h < height / 2 {
        Some(Rectangle::new(min_x, min_y, w, h))
    } else {
        None
    }
}
