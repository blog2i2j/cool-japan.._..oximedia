//! Advanced inpainting algorithms for delogo filter.
//!
//! Contains Criminisi's exemplar-based inpainting and Telea's fast marching inpainting.

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(dead_code)]

/// Criminisi's exemplar-based inpainting.
pub struct CriminisiInpainting {
    /// Patch size.
    patch_size: usize,
    /// Priority weight for data term.
    alpha: f32,
    /// Priority weight for confidence term.
    beta: f32,
}

impl CriminisiInpainting {
    /// Create a new Criminisi inpainter.
    #[must_use]
    pub fn new(patch_size: usize) -> Self {
        Self {
            patch_size,
            alpha: 0.7,
            beta: 0.3,
        }
    }

    /// Compute fill priority for a pixel on the boundary.
    #[must_use]
    pub fn compute_priority(
        &self,
        data: &[u8],
        mask: &[bool],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> f32 {
        let confidence = self.compute_confidence(mask, x, y, width, height);
        let data_term = self.compute_data_term(data, mask, x, y, width, height);

        self.beta * confidence + self.alpha * data_term
    }

    /// Compute confidence term (ratio of known pixels in patch).
    fn compute_confidence(&self, mask: &[bool], x: u32, y: u32, width: u32, height: u32) -> f32 {
        let half = self.patch_size as i32 / 2;
        let mut known = 0;
        let mut total = 0;

        for dy in -half..=half {
            for dx in -half..=half {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    total += 1;
                    let idx = (ny as u32 * width + nx as u32) as usize;
                    if !mask.get(idx).copied().unwrap_or(false) {
                        known += 1;
                    }
                }
            }
        }

        if total > 0 {
            known as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Compute data term (gradient strength perpendicular to boundary).
    fn compute_data_term(
        &self,
        data: &[u8],
        _mask: &[bool],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> f32 {
        // Compute image gradient
        let (gx, gy) = self.compute_gradient(data, x, y, width, height);
        let magnitude = (gx * gx + gy * gy).sqrt();

        magnitude / 255.0
    }

    /// Compute image gradient.
    fn compute_gradient(&self, data: &[u8], x: u32, y: u32, width: u32, height: u32) -> (f32, f32) {
        let mut gx = 0.0f32;
        let mut gy = 0.0f32;

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nidx = (ny as u32 * width + nx as u32) as usize;
                    let val = data.get(nidx).copied().unwrap_or(128) as f32;

                    gx += val * dx as f32;
                    gy += val * dy as f32;
                }
            }
        }

        (gx, gy)
    }
}

/// Telea's fast marching inpainting.
pub struct TeleaInpainting {
    /// Neighborhood radius.
    radius: usize,
}

impl TeleaInpainting {
    /// Create a new Telea inpainter.
    #[must_use]
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Inpaint a pixel using weighted average of known neighbors.
    #[must_use]
    pub fn inpaint_pixel(
        &self,
        data: &[u8],
        mask: &[bool],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> u8 {
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        let r = self.radius as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nidx = (ny as u32 * width + nx as u32) as usize;

                    if !mask.get(nidx).copied().unwrap_or(false) {
                        let dist = ((dx * dx + dy * dy) as f32).sqrt();
                        let weight = 1.0 / (dist + 0.1);

                        sum += data.get(nidx).copied().unwrap_or(128) as f32 * weight;
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
}
