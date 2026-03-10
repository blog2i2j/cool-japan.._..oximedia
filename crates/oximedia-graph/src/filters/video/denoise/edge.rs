// Edge detection utilities for edge-preserving filtering.

/// Detect edges using Sobel operator.
#[must_use]
pub fn detect_sobel(data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let mut edges = vec![0.0f32; (width * height) as usize];

    let sobel_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    let sobel_y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;

            for ky in 0..3 {
                for kx in 0..3 {
                    let nx = x + kx - 1;
                    let ny = y + ky - 1;
                    let nidx = (ny * width + nx) as usize;
                    let val = data.get(nidx).copied().unwrap_or(128) as f32;

                    let kidx = (ky * 3 + kx) as usize;
                    gx += val * sobel_x[kidx] as f32;
                    gy += val * sobel_y[kidx] as f32;
                }
            }

            let magnitude = (gx * gx + gy * gy).sqrt();
            edges[(y * width + x) as usize] = magnitude;
        }
    }

    edges
}

/// Detect edges using Laplacian operator.
#[must_use]
pub fn detect_laplacian(data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let mut edges = vec![0.0f32; (width * height) as usize];

    let laplacian = [0, -1, 0, -1, 4, -1, 0, -1, 0];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0.0f32;

            for ky in 0..3 {
                for kx in 0..3 {
                    let nx = x + kx - 1;
                    let ny = y + ky - 1;
                    let nidx = (ny * width + nx) as usize;
                    let val = data.get(nidx).copied().unwrap_or(128) as f32;

                    let kidx = (ky * 3 + kx) as usize;
                    sum += val * laplacian[kidx] as f32;
                }
            }

            edges[(y * width + x) as usize] = sum.abs();
        }
    }

    edges
}

/// Compute edge map with threshold.
#[must_use]
pub fn edge_map(edges: &[f32], threshold: f32) -> Vec<bool> {
    edges.iter().map(|&e| e > threshold).collect()
}

/// Non-maximum suppression for edge thinning.
pub fn non_maximum_suppression(edges: &mut [f32], width: u32, height: u32) {
    let original = edges.to_vec();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            let val = original[idx];

            let neighbors = [
                original.get(idx - 1).copied().unwrap_or(0.0),
                original.get(idx + 1).copied().unwrap_or(0.0),
                original
                    .get((idx as u32 - width) as usize)
                    .copied()
                    .unwrap_or(0.0),
                original
                    .get((idx as u32 + width) as usize)
                    .copied()
                    .unwrap_or(0.0),
            ];

            if !neighbors.iter().all(|&n| val >= n) {
                edges[idx] = 0.0;
            }
        }
    }
}
