//! Boundary handling modes for frame warping.

use scirs2_core::ndarray::Array2;

/// Boundary handling mode for pixels outside the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Constant value (black)
    Constant,
    /// Replicate edge pixels
    Replicate,
    /// Reflect across edge
    Reflect,
    /// Wrap around
    Wrap,
}

impl BoundaryMode {
    /// Get pixel value with boundary handling.
    #[must_use]
    pub fn get_pixel(self, data: &Array2<u8>, x: f64, y: f64) -> u8 {
        let (height, width) = data.dim();
        let (x, y) = self.handle_bounds(x, y, width, height);

        if x < 0.0 || y < 0.0 || x >= width as f64 || y >= height as f64 {
            return 0; // Constant mode fallback
        }

        let xi = x as usize;
        let yi = y as usize;

        data[[yi.min(height - 1), xi.min(width - 1)]]
    }

    /// Handle out-of-bounds coordinates.
    fn handle_bounds(self, x: f64, y: f64, width: usize, height: usize) -> (f64, f64) {
        let w = width as f64;
        let h = height as f64;

        match self {
            Self::Constant => (x, y),
            Self::Replicate => (x.clamp(0.0, w - 1.0), y.clamp(0.0, h - 1.0)),
            Self::Reflect => {
                let x = if x < 0.0 {
                    -x
                } else if x >= w {
                    2.0 * w - x - 1.0
                } else {
                    x
                };
                let y = if y < 0.0 {
                    -y
                } else if y >= h {
                    2.0 * h - y - 1.0
                } else {
                    y
                };
                (x, y)
            }
            Self::Wrap => {
                let x = if x < 0.0 {
                    x + w
                } else if x >= w {
                    x - w
                } else {
                    x
                };
                let y = if y < 0.0 {
                    y + h
                } else if y >= h {
                    y - h
                } else {
                    y
                };
                (x, y)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_modes() {
        let data = Array2::from_elem((10, 10), 128);

        let _ = BoundaryMode::Constant.get_pixel(&data, 5.0, 5.0);
        let _ = BoundaryMode::Replicate.get_pixel(&data, -1.0, -1.0);
        let _ = BoundaryMode::Reflect.get_pixel(&data, 11.0, 11.0);
        let _ = BoundaryMode::Wrap.get_pixel(&data, 11.0, 11.0);
    }
}
