//! Mask generation utilities for delogo filter.

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use super::Rectangle;

/// Mask type for logo regions.
#[derive(Clone, Debug)]
pub struct LogoMask {
    /// Width of the mask.
    pub width: u32,
    /// Height of the mask.
    pub height: u32,
    /// Mask data (0.0 = keep original, 1.0 = fully remove).
    pub data: Vec<f32>,
}

impl LogoMask {
    /// Create a new mask from a rectangle.
    #[must_use]
    pub fn from_rectangle(region: &Rectangle, width: u32, height: u32) -> Self {
        let mut data = vec![0.0f32; (width * height) as usize];

        for y in region.y..region.bottom().min(height) {
            for x in region.x..region.right().min(width) {
                data[(y * width + x) as usize] = 1.0;
            }
        }

        Self {
            width,
            height,
            data,
        }
    }

    /// Create a feathered mask.
    #[must_use]
    pub fn from_rectangle_feathered(
        region: &Rectangle,
        width: u32,
        height: u32,
        feather: u32,
    ) -> Self {
        let mut data = vec![0.0f32; (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let dist = distance_to_rectangle(x, y, region);
                let alpha = if dist < 0.0 {
                    // Inside
                    1.0
                } else if dist < feather as f32 {
                    // Feather zone
                    1.0 - (dist / feather as f32)
                } else {
                    // Outside
                    0.0
                };

                data[(y * width + x) as usize] = alpha;
            }
        }

        Self {
            width,
            height,
            data,
        }
    }

    /// Get mask value at position.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.data
            .get((y * self.width + x) as usize)
            .copied()
            .unwrap_or(0.0)
    }

    /// Blur the mask for smoother transitions.
    pub fn blur(&mut self, radius: usize) {
        let kernel = super::create_gaussian_kernel(radius, radius as f32 / 2.0);
        let ksize = radius * 2 + 1;

        let mut blurred = self.data.clone();

        for y in 0..self.height {
            for x in 0..self.width {
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for ky in 0..ksize {
                    let py = y as i32 + ky as i32 - radius as i32;
                    if py < 0 || py >= self.height as i32 {
                        continue;
                    }

                    for kx in 0..ksize {
                        let px = x as i32 + kx as i32 - radius as i32;
                        if px < 0 || px >= self.width as i32 {
                            continue;
                        }

                        let idx = (py as u32 * self.width + px as u32) as usize;
                        let weight = kernel[ky * ksize + kx];
                        sum += self.data.get(idx).copied().unwrap_or(0.0) * weight;
                        weight_sum += weight;
                    }
                }

                let idx = (y * self.width + x) as usize;
                if weight_sum > 0.0 {
                    blurred[idx] = sum / weight_sum;
                }
            }
        }

        self.data = blurred;
    }

    /// Dilate the mask (expand regions).
    pub fn dilate(&mut self, iterations: usize) {
        for _ in 0..iterations {
            let mut dilated = self.data.clone();

            for y in 1..self.height - 1 {
                for x in 1..self.width - 1 {
                    let mut max_val = self.get(x, y);

                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = (x as i32 + dx) as u32;
                            let ny = (y as i32 + dy) as u32;
                            max_val = max_val.max(self.get(nx, ny));
                        }
                    }

                    dilated[(y * self.width + x) as usize] = max_val;
                }
            }

            self.data = dilated;
        }
    }

    /// Erode the mask (shrink regions).
    pub fn erode(&mut self, iterations: usize) {
        for _ in 0..iterations {
            let mut eroded = self.data.clone();

            for y in 1..self.height - 1 {
                for x in 1..self.width - 1 {
                    let mut min_val = self.get(x, y);

                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = (x as i32 + dx) as u32;
                            let ny = (y as i32 + dy) as u32;
                            min_val = min_val.min(self.get(nx, ny));
                        }
                    }

                    eroded[(y * self.width + x) as usize] = min_val;
                }
            }

            self.data = eroded;
        }
    }
}

/// Compute signed distance from point to rectangle.
pub fn distance_to_rectangle(x: u32, y: u32, region: &Rectangle) -> f32 {
    if region.contains(x, y) {
        // Inside - negative distance to nearest edge
        let dx = (x - region.x).min(region.right() - x - 1);
        let dy = (y - region.y).min(region.bottom() - y - 1);
        -(dx.min(dy) as f32)
    } else {
        // Outside - distance to nearest edge
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
    }
}
