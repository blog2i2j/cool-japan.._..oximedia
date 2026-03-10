// Quality metrics for evaluating denoise results.

/// Compute PSNR between original and denoised frames.
#[must_use]
pub fn psnr(original: &[u8], denoised: &[u8]) -> f32 {
    if original.len() != denoised.len() || original.is_empty() {
        return 0.0;
    }

    let mut mse = 0.0f64;
    for (a, b) in original.iter().zip(denoised.iter()) {
        let diff = *a as f64 - *b as f64;
        mse += diff * diff;
    }

    mse /= original.len() as f64;

    if mse < 1e-10 {
        return f32::INFINITY;
    }

    let max_val = 255.0;
    (10.0 * (max_val * max_val / mse).log10()) as f32
}

/// Compute SNR (Signal-to-Noise Ratio).
#[must_use]
pub fn snr(original: &[u8], denoised: &[u8]) -> f32 {
    if original.len() != denoised.len() || original.is_empty() {
        return 0.0;
    }

    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;

    for (a, b) in original.iter().zip(denoised.iter()) {
        let signal = *a as f64;
        let noise = (*a as f64 - *b as f64).abs();

        signal_power += signal * signal;
        noise_power += noise * noise;
    }

    if noise_power < 1e-10 {
        return f32::INFINITY;
    }

    (10.0 * (signal_power / noise_power).log10()) as f32
}

/// Compute mean absolute error.
#[must_use]
pub fn mae(original: &[u8], denoised: &[u8]) -> f32 {
    if original.len() != denoised.len() || original.is_empty() {
        return 0.0;
    }

    let sum: f32 = original
        .iter()
        .zip(denoised.iter())
        .map(|(a, b)| (*a as f32 - *b as f32).abs())
        .sum();

    sum / original.len() as f32
}

/// Compute structural similarity index (SSIM).
#[must_use]
pub fn ssim(original: &[u8], denoised: &[u8], width: usize, height: usize) -> f32 {
    if original.len() != denoised.len() || original.is_empty() {
        return 0.0;
    }

    let c1 = (0.01 * 255.0) * (0.01 * 255.0);
    let c2 = (0.03 * 255.0) * (0.03 * 255.0);

    let window_size = 11usize;
    let half = window_size / 2;

    let mut ssim_sum = 0.0f64;
    let mut count = 0;

    for y in half..(height - half) {
        for x in half..(width - half) {
            let (mean_x, mean_y, var_x, var_y, cov_xy) =
                compute_window_stats(original, denoised, x, y, width, window_size);

            let ssim_val = ((2.0 * mean_x * mean_y + c1) * (2.0 * cov_xy + c2))
                / ((mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2));

            ssim_sum += ssim_val;
            count += 1;
        }
    }

    if count > 0 {
        (ssim_sum / count as f64) as f32
    } else {
        0.0
    }
}

fn compute_window_stats(
    img1: &[u8],
    img2: &[u8],
    cx: usize,
    cy: usize,
    width: usize,
    window_size: usize,
) -> (f64, f64, f64, f64, f64) {
    let half = window_size / 2;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_yy = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut count = 0;

    for dy in 0..window_size {
        let y = cy + dy - half;
        for dx in 0..window_size {
            let x = cx + dx - half;
            let idx = y * width + x;

            let val_x = img1.get(idx).copied().unwrap_or(0) as f64;
            let val_y = img2.get(idx).copied().unwrap_or(0) as f64;

            sum_x += val_x;
            sum_y += val_y;
            sum_xx += val_x * val_x;
            sum_yy += val_y * val_y;
            sum_xy += val_x * val_y;
            count += 1;
        }
    }

    let n = count as f64;
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let var_x = sum_xx / n - mean_x * mean_x;
    let var_y = sum_yy / n - mean_y * mean_y;
    let cov_xy = sum_xy / n - mean_x * mean_y;

    (mean_x, mean_y, var_x, var_y, cov_xy)
}
