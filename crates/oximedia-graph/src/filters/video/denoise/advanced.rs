// Advanced filtering techniques.

/// Wiener filter for noise reduction.
#[derive(Debug, Clone)]
pub struct WienerFilter {
    /// Noise variance estimate.
    noise_variance: f32,
    /// Window size.
    window_size: u32,
}

impl WienerFilter {
    /// Create a new Wiener filter.
    #[must_use]
    pub fn new(noise_variance: f32, window_size: u32) -> Self {
        Self {
            noise_variance,
            window_size,
        }
    }

    /// Apply Wiener filter to data.
    pub fn apply(&self, data: &mut [u8], width: u32, height: u32) {
        let original = data.to_vec();
        let radius = self.window_size / 2;

        for y in 0..height {
            for x in 0..width {
                let (mean, variance) = compute_window_stats(&original, x, y, width, height, radius);

                let idx = (y * width + x) as usize;
                let pixel = original.get(idx).copied().unwrap_or(128) as f32;

                let local_variance = variance.max(0.0);
                let weight = if local_variance > self.noise_variance {
                    1.0 - self.noise_variance / local_variance
                } else {
                    0.0
                };

                let filtered = mean + weight * (pixel - mean);
                data[idx] = filtered.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn compute_window_stats(
    data: &[u8],
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    radius: u32,
) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut sq_sum = 0.0f32;
    let mut count = 0;

    let y_min = y.saturating_sub(radius);
    let y_max = (y + radius + 1).min(height);
    let x_min = x.saturating_sub(radius);
    let x_max = (x + radius + 1).min(width);

    for ny in y_min..y_max {
        for nx in x_min..x_max {
            let nidx = (ny * width + nx) as usize;
            let val = data.get(nidx).copied().unwrap_or(128) as f32;
            sum += val;
            sq_sum += val * val;
            count += 1;
        }
    }

    if count > 0 {
        let mean = sum / count as f32;
        let variance = (sq_sum / count as f32) - (mean * mean);
        (mean, variance)
    } else {
        (128.0, 0.0)
    }
}

/// Anisotropic diffusion for edge-preserving smoothing.
#[derive(Debug, Clone)]
pub struct AnisotropicDiffusion {
    /// Number of iterations.
    iterations: u32,
    /// Diffusion coefficient.
    kappa: f32,
    /// Time step.
    delta_t: f32,
}

impl AnisotropicDiffusion {
    /// Create a new anisotropic diffusion filter.
    #[must_use]
    pub fn new(iterations: u32, kappa: f32) -> Self {
        Self {
            iterations,
            kappa,
            delta_t: 0.25,
        }
    }

    /// Apply anisotropic diffusion.
    pub fn apply(&self, data: &mut [u8], width: u32, height: u32) {
        let kappa_sq = self.kappa * self.kappa;

        for _ in 0..self.iterations {
            let original = data.to_vec();

            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let idx = (y * width + x) as usize;
                    let center = original.get(idx).copied().unwrap_or(128) as f32;

                    let north = original
                        .get(((y - 1) * width + x) as usize)
                        .copied()
                        .unwrap_or(128) as f32;
                    let south = original
                        .get(((y + 1) * width + x) as usize)
                        .copied()
                        .unwrap_or(128) as f32;
                    let west = original
                        .get((y * width + (x - 1)) as usize)
                        .copied()
                        .unwrap_or(128) as f32;
                    let east = original
                        .get((y * width + (x + 1)) as usize)
                        .copied()
                        .unwrap_or(128) as f32;

                    let grad_n = north - center;
                    let grad_s = south - center;
                    let grad_w = west - center;
                    let grad_e = east - center;

                    let c_n = self.diffusion_coeff(grad_n, kappa_sq);
                    let c_s = self.diffusion_coeff(grad_s, kappa_sq);
                    let c_w = self.diffusion_coeff(grad_w, kappa_sq);
                    let c_e = self.diffusion_coeff(grad_e, kappa_sq);

                    let update =
                        self.delta_t * (c_n * grad_n + c_s * grad_s + c_w * grad_w + c_e * grad_e);

                    data[idx] = (center + update).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    fn diffusion_coeff(&self, gradient: f32, kappa_sq: f32) -> f32 {
        let grad_sq = gradient * gradient;
        (-(grad_sq / kappa_sq)).exp()
    }
}

/// Total variation denoising.
#[derive(Debug, Clone)]
pub struct TotalVariationDenoising {
    /// Regularization parameter.
    lambda: f32,
    /// Number of iterations.
    iterations: u32,
}

impl TotalVariationDenoising {
    /// Create a new TV denoising filter.
    #[must_use]
    pub fn new(lambda: f32, iterations: u32) -> Self {
        Self { lambda, iterations }
    }

    /// Apply TV denoising.
    pub fn apply(&self, data: &mut [u8], width: u32, height: u32) {
        let noisy = data.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let mut denoised = noisy.clone();

        for _ in 0..self.iterations {
            let prev = denoised.clone();

            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let idx = (y * width + x) as usize;

                    let grad_x = prev
                        .get((y * width + (x + 1)) as usize)
                        .copied()
                        .unwrap_or(128.0)
                        - prev
                            .get((y * width + (x - 1)) as usize)
                            .copied()
                            .unwrap_or(128.0);

                    let grad_y = prev
                        .get(((y + 1) * width + x) as usize)
                        .copied()
                        .unwrap_or(128.0)
                        - prev
                            .get(((y - 1) * width + x) as usize)
                            .copied()
                            .unwrap_or(128.0);

                    let grad_mag = (grad_x * grad_x + grad_y * grad_y).sqrt() + 1e-8;

                    let div_x = (prev
                        .get((y * width + (x + 1)) as usize)
                        .copied()
                        .unwrap_or(128.0)
                        - 2.0 * prev[idx]
                        + prev
                            .get((y * width + (x - 1)) as usize)
                            .copied()
                            .unwrap_or(128.0))
                        / grad_mag;

                    let div_y = (prev
                        .get(((y + 1) * width + x) as usize)
                        .copied()
                        .unwrap_or(128.0)
                        - 2.0 * prev[idx]
                        + prev
                            .get(((y - 1) * width + x) as usize)
                            .copied()
                            .unwrap_or(128.0))
                        / grad_mag;

                    let divergence = div_x + div_y;
                    let data_term = noisy[idx] - denoised[idx];

                    denoised[idx] += 0.1 * (data_term / self.lambda + divergence);
                }
            }
        }

        for (i, &val) in denoised.iter().enumerate() {
            data[i] = val.round().clamp(0.0, 255.0) as u8;
        }
    }
}
