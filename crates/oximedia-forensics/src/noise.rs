//! Noise Pattern Analysis and PRNU Detection
//!
//! This module analyzes noise patterns in images to detect tampering.
//! Photo Response Non-Uniformity (PRNU) is a unique sensor fingerprint
//! that can be used to detect splicing and cloning.

use crate::{ForensicTest, ForensicsError, ForensicsResult};
use image::RgbImage;
use ndarray::{Array2, Array3};

/// PRNU noise pattern
#[derive(Debug, Clone)]
pub struct PrnuPattern {
    /// Noise pattern for each color channel
    pub pattern: Array3<f64>,
    /// Pattern strength
    pub strength: f64,
}

/// Noise inconsistency result
#[derive(Debug, Clone)]
pub struct NoiseInconsistency {
    /// Regions with inconsistent noise
    pub inconsistent_regions: Vec<(usize, usize, usize, usize)>,
    /// Inconsistency score
    pub score: f64,
    /// Confidence
    pub confidence: f64,
}

/// Analyze noise patterns for tampering detection
#[allow(unused_variables)]
pub fn analyze_noise(image: &RgbImage) -> ForensicsResult<ForensicTest> {
    let mut test = ForensicTest::new("Noise Pattern Analysis");

    let (width, height) = image.dimensions();

    // Extract PRNU pattern
    let prnu = extract_prnu_pattern(image)?;
    test.add_finding(format!(
        "PRNU pattern extracted with strength: {:.4}",
        prnu.strength
    ));

    // Analyze noise consistency across regions
    let inconsistency = detect_noise_inconsistency(image)?;

    if inconsistency.score > 0.3 {
        test.tampering_detected = true;
        test.add_finding(format!(
            "Noise inconsistency detected (score: {:.3})",
            inconsistency.score
        ));
        test.add_finding(format!(
            "Found {} regions with inconsistent noise patterns",
            inconsistency.inconsistent_regions.len()
        ));
    }

    // Detect cloning/copy-paste via noise correlation
    let cloning_detected = detect_cloning_via_noise(image)?;
    if cloning_detected {
        test.tampering_detected = true;
        test.add_finding("Potential cloning detected via noise correlation".to_string());
    }

    // Calculate overall confidence
    let mut confidence = inconsistency.confidence;
    if cloning_detected {
        confidence = (confidence + 0.3).min(1.0);
    }

    test.set_confidence(confidence);

    // Create anomaly map from noise inconsistency
    let anomaly_map = create_noise_anomaly_map(image, &inconsistency)?;
    test.anomaly_map = Some(anomaly_map);

    Ok(test)
}

/// Extract PRNU pattern from image
fn extract_prnu_pattern(image: &RgbImage) -> ForensicsResult<PrnuPattern> {
    let (width, height) = image.dimensions();

    // Convert to float arrays
    let mut r_channel = Array2::zeros((height as usize, width as usize));
    let mut g_channel = Array2::zeros((height as usize, width as usize));
    let mut b_channel = Array2::zeros((height as usize, width as usize));

    for (x, y, pixel) in image.enumerate_pixels() {
        r_channel[[y as usize, x as usize]] = pixel[0] as f64;
        g_channel[[y as usize, x as usize]] = pixel[1] as f64;
        b_channel[[y as usize, x as usize]] = pixel[2] as f64;
    }

    // Denoise using Wiener filter approximation
    let r_denoised = wiener_filter(&r_channel);
    let g_denoised = wiener_filter(&g_channel);
    let b_denoised = wiener_filter(&b_channel);

    // Extract noise residual (PRNU)
    let mut pattern = Array3::zeros((height as usize, width as usize, 3));
    let mut strength_sum = 0.0;
    let mut count = 0;

    for y in 0..height as usize {
        for x in 0..width as usize {
            let r_noise = r_channel[[y, x]] - r_denoised[[y, x]];
            let g_noise = g_channel[[y, x]] - g_denoised[[y, x]];
            let b_noise = b_channel[[y, x]] - b_denoised[[y, x]];

            pattern[[y, x, 0]] = r_noise;
            pattern[[y, x, 1]] = g_noise;
            pattern[[y, x, 2]] = b_noise;

            strength_sum += (r_noise * r_noise + g_noise * g_noise + b_noise * b_noise).sqrt();
            count += 1;
        }
    }

    let strength = if count > 0 {
        strength_sum / count as f64
    } else {
        0.0
    };

    Ok(PrnuPattern { pattern, strength })
}

/// Wiener filter for denoising (simplified version)
fn wiener_filter(channel: &Array2<f64>) -> Array2<f64> {
    let (height, width) = channel.dim();
    let mut filtered = channel.clone();

    let kernel_size = 5;
    let half_kernel = kernel_size / 2;

    for y in half_kernel..height - half_kernel {
        for x in half_kernel..width - half_kernel {
            let mut sum = 0.0;
            let mut count = 0;

            // Local mean
            for dy in -(half_kernel as i32)..=half_kernel as i32 {
                for dx in -(half_kernel as i32)..=half_kernel as i32 {
                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;
                    sum += channel[[ny, nx]];
                    count += 1;
                }
            }

            let local_mean = sum / count as f64;

            // Local variance
            let mut var_sum = 0.0;
            for dy in -(half_kernel as i32)..=half_kernel as i32 {
                for dx in -(half_kernel as i32)..=half_kernel as i32 {
                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;
                    let diff = channel[[ny, nx]] - local_mean;
                    var_sum += diff * diff;
                }
            }

            let local_variance = var_sum / count as f64;

            // Wiener filter coefficient
            let noise_variance = 10.0; // Estimated noise variance
            let wiener_coeff = if local_variance > 0.0 {
                ((local_variance - noise_variance) / local_variance).max(0.0)
            } else {
                0.0
            };

            filtered[[y, x]] = local_mean + wiener_coeff * (channel[[y, x]] - local_mean);
        }
    }

    filtered
}

/// Detect noise inconsistency across image regions
fn detect_noise_inconsistency(image: &RgbImage) -> ForensicsResult<NoiseInconsistency> {
    let (width, height) = image.dimensions();
    let region_size = 64;

    // Extract noise from regions
    let mut region_noise_levels = Vec::new();

    for y in (0..height).step_by(region_size) {
        for x in (0..width).step_by(region_size) {
            let noise_level = calculate_region_noise(image, x, y, region_size as u32);
            region_noise_levels.push((x, y, noise_level));
        }
    }

    // Calculate mean and std dev of noise levels
    let mean_noise: f64 = region_noise_levels.iter().map(|(_, _, n)| n).sum::<f64>()
        / region_noise_levels.len() as f64;

    let variance: f64 = region_noise_levels
        .iter()
        .map(|(_, _, n)| {
            let diff = n - mean_noise;
            diff * diff
        })
        .sum::<f64>()
        / region_noise_levels.len() as f64;

    let std_dev = variance.sqrt();

    // Find regions with significantly different noise
    let threshold = mean_noise + 2.0 * std_dev;
    let mut inconsistent_regions = Vec::new();

    for (x, y, noise_level) in &region_noise_levels {
        if (noise_level - mean_noise).abs() > threshold || *noise_level < mean_noise * 0.5 {
            inconsistent_regions.push((*x as usize, *y as usize, region_size, region_size));
        }
    }

    let score = if !region_noise_levels.is_empty() {
        (inconsistent_regions.len() as f64 / region_noise_levels.len() as f64).min(1.0)
    } else {
        0.0
    };

    let confidence = if std_dev > 0.0 {
        (std_dev / mean_noise).min(1.0)
    } else {
        0.0
    };

    Ok(NoiseInconsistency {
        inconsistent_regions,
        score,
        confidence,
    })
}

/// Calculate noise level in a region
fn calculate_region_noise(image: &RgbImage, x: u32, y: u32, size: u32) -> f64 {
    let (width, height) = image.dimensions();

    let mut noise_sum = 0.0;
    let mut count = 0;

    for dy in 0..size {
        for dx in 0..size {
            let px = x + dx;
            let py = y + dy;

            if px < width && py < height {
                // Use Laplacian for noise estimation
                if px > 0 && px < width - 1 && py > 0 && py < height - 1 {
                    let center = image.get_pixel(px, py);
                    let left = image.get_pixel(px - 1, py);
                    let right = image.get_pixel(px + 1, py);
                    let top = image.get_pixel(px, py - 1);
                    let bottom = image.get_pixel(px, py + 1);

                    for c in 0..3 {
                        let laplacian = (center[c] as f64) * 4.0
                            - (left[c] as f64)
                            - (right[c] as f64)
                            - (top[c] as f64)
                            - (bottom[c] as f64);

                        noise_sum += laplacian.abs();
                        count += 1;
                    }
                }
            }
        }
    }

    if count > 0 {
        noise_sum / count as f64
    } else {
        0.0
    }
}

/// Detect cloning/splicing via noise pattern correlation
fn detect_cloning_via_noise(image: &RgbImage) -> ForensicsResult<bool> {
    let prnu = extract_prnu_pattern(image)?;
    let (height, width, _) = prnu.pattern.dim();

    let patch_size = 32;
    let step = 16;

    // Extract patches and compute correlations
    let mut max_correlation: f64 = 0.0;

    for y1 in (0..height - patch_size).step_by(step) {
        for x1 in (0..width - patch_size).step_by(step) {
            for y2 in (y1 + patch_size..height - patch_size).step_by(step) {
                for x2 in (0..width - patch_size).step_by(step) {
                    let correlation =
                        compute_patch_correlation(&prnu.pattern, x1, y1, x2, y2, patch_size);

                    max_correlation = max_correlation.max(correlation);

                    // Early exit if high correlation found
                    if max_correlation > 0.9 {
                        return Ok(true);
                    }
                }
            }
        }
    }

    // Threshold for cloning detection
    Ok(max_correlation > 0.85)
}

/// Compute correlation between two noise patches
fn compute_patch_correlation(
    pattern: &Array3<f64>,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    size: usize,
) -> f64 {
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;
    let mut sum_prod = 0.0;
    let mut count = 0;

    for dy in 0..size {
        for dx in 0..size {
            for c in 0..3 {
                let val1 = pattern[[y1 + dy, x1 + dx, c]];
                let val2 = pattern[[y2 + dy, x2 + dx, c]];

                sum1 += val1;
                sum2 += val2;
                sum_sq1 += val1 * val1;
                sum_sq2 += val2 * val2;
                sum_prod += val1 * val2;
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let n = count as f64;
    let mean1 = sum1 / n;
    let mean2 = sum2 / n;

    let var1 = sum_sq1 / n - mean1 * mean1;
    let var2 = sum_sq2 / n - mean2 * mean2;
    let covar = sum_prod / n - mean1 * mean2;

    if var1 > 0.0 && var2 > 0.0 {
        covar / (var1.sqrt() * var2.sqrt())
    } else {
        0.0
    }
}

/// Create anomaly map from noise inconsistency
fn create_noise_anomaly_map(
    image: &RgbImage,
    inconsistency: &NoiseInconsistency,
) -> ForensicsResult<Array2<f64>> {
    let (width, height) = image.dimensions();
    let mut anomaly_map = Array2::zeros((height as usize, width as usize));

    // Mark inconsistent regions
    for (x, y, w, h) in &inconsistency.inconsistent_regions {
        for dy in 0..*h {
            for dx in 0..*w {
                let px = x + dx;
                let py = y + dy;

                if px < width as usize && py < height as usize {
                    anomaly_map[[py, px]] = 1.0;
                }
            }
        }
    }

    Ok(anomaly_map)
}

/// Advanced PRNU-based splicing detection
pub fn detect_splicing_prnu(
    image: &RgbImage,
) -> ForensicsResult<Vec<(usize, usize, usize, usize)>> {
    let prnu = extract_prnu_pattern(image)?;
    let (height, width, _) = prnu.pattern.dim();

    let region_size = 64;
    let mut spliced_regions = Vec::new();

    // Analyze PRNU consistency across regions
    let mut region_prnu_strengths = Vec::new();

    for y in (0..height - region_size).step_by(region_size / 2) {
        for x in (0..width - region_size).step_by(region_size / 2) {
            let strength = compute_region_prnu_strength(&prnu.pattern, x, y, region_size);
            region_prnu_strengths.push((x, y, strength));
        }
    }

    // Find outliers
    let mean_strength: f64 = region_prnu_strengths.iter().map(|(_, _, s)| s).sum::<f64>()
        / region_prnu_strengths.len() as f64;

    let variance: f64 = region_prnu_strengths
        .iter()
        .map(|(_, _, s)| {
            let diff = s - mean_strength;
            diff * diff
        })
        .sum::<f64>()
        / region_prnu_strengths.len() as f64;

    let std_dev = variance.sqrt();
    let threshold = 2.0 * std_dev;

    for (x, y, strength) in region_prnu_strengths {
        if (strength - mean_strength).abs() > threshold {
            spliced_regions.push((x, y, region_size, region_size));
        }
    }

    Ok(spliced_regions)
}

/// Compute PRNU strength in a region
fn compute_region_prnu_strength(pattern: &Array3<f64>, x: usize, y: usize, size: usize) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;

    for dy in 0..size {
        for dx in 0..size {
            for c in 0..3 {
                if y + dy < pattern.shape()[0] && x + dx < pattern.shape()[1] {
                    let val = pattern[[y + dy, x + dx, c]];
                    sum += val * val;
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        (sum / count as f64).sqrt()
    } else {
        0.0
    }
}

/// Extract sensor fingerprint from multiple images
pub fn extract_sensor_fingerprint(images: &[RgbImage]) -> ForensicsResult<PrnuPattern> {
    if images.is_empty() {
        return Err(ForensicsError::InvalidImage(
            "No images provided".to_string(),
        ));
    }

    // Extract PRNU from each image and average
    let prnu_patterns: Vec<PrnuPattern> = images
        .iter()
        .map(extract_prnu_pattern)
        .collect::<Result<Vec<_>, _>>()?;

    let (height, width, _) = prnu_patterns[0].pattern.dim();
    let mut averaged_pattern = Array3::zeros((height, width, 3));

    for prnu in &prnu_patterns {
        averaged_pattern += &prnu.pattern;
    }

    averaged_pattern /= prnu_patterns.len() as f64;

    let strength =
        prnu_patterns.iter().map(|p| p.strength).sum::<f64>() / prnu_patterns.len() as f64;

    Ok(PrnuPattern {
        pattern: averaged_pattern,
        strength,
    })
}

/// Verify image authenticity using known sensor fingerprint
pub fn verify_with_fingerprint(
    image: &RgbImage,
    fingerprint: &PrnuPattern,
) -> ForensicsResult<f64> {
    let image_prnu = extract_prnu_pattern(image)?;

    // Compute correlation between image PRNU and known fingerprint
    let correlation = compute_pattern_correlation(&image_prnu.pattern, &fingerprint.pattern);

    Ok(correlation)
}

/// Compute correlation between two PRNU patterns
fn compute_pattern_correlation(pattern1: &Array3<f64>, pattern2: &Array3<f64>) -> f64 {
    let (height, width, channels) = pattern1.dim();

    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;
    let mut sum_prod = 0.0;
    let mut count = 0;

    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let val1 = pattern1[[y, x, c]];
                let val2 = pattern2[[y, x, c]];

                sum1 += val1;
                sum2 += val2;
                sum_sq1 += val1 * val1;
                sum_sq2 += val2 * val2;
                sum_prod += val1 * val2;
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let n = count as f64;
    let mean1 = sum1 / n;
    let mean2 = sum2 / n;

    let var1 = sum_sq1 / n - mean1 * mean1;
    let var2 = sum_sq2 / n - mean2 * mean2;
    let covar = sum_prod / n - mean1 * mean2;

    if var1 > 0.0 && var2 > 0.0 {
        covar / (var1.sqrt() * var2.sqrt())
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_prnu_extraction() {
        let img = RgbImage::new(64, 64);
        let result = extract_prnu_pattern(&img);
        assert!(result.is_ok());
        let prnu = result.expect("prnu should be valid");
        assert_eq!(prnu.pattern.dim(), (64, 64, 3));
    }

    #[test]
    fn test_region_noise_calculation() {
        let img = RgbImage::new(128, 128);
        let noise = calculate_region_noise(&img, 0, 0, 32);
        assert!(noise >= 0.0);
    }

    #[test]
    fn test_wiener_filter() {
        let channel = Array2::from_elem((10, 10), 100.0);
        let filtered = wiener_filter(&channel);
        assert_eq!(filtered.dim(), channel.dim());
    }

    #[test]
    fn test_patch_correlation() {
        let pattern = Array3::zeros((64, 64, 3));
        let corr = compute_patch_correlation(&pattern, 0, 0, 16, 16, 8);
        assert!(corr >= -1.0 && corr <= 1.0);
    }

    #[test]
    fn test_noise_inconsistency_detection() {
        let img = RgbImage::new(128, 128);
        let result = detect_noise_inconsistency(&img);
        assert!(result.is_ok());
    }
}
