//! Color analysis and palette extraction.
//!
//! This module analyzes color content in video:
//! - **Dominant Colors** - K-means clustering for palette extraction
//! - **Color Distribution** - RGB histograms and statistics
//! - **Color Grading** - Detect color grading and LUT applications
//! - **Saturation Analysis** - Color intensity metrics
//!
//! # Algorithms
//!
//! - K-means clustering for dominant color extraction
//! - YUV to RGB conversion for color analysis
//! - Histogram-based color distribution

use crate::{AnalysisError, AnalysisResult};
use serde::{Deserialize, Serialize};

/// Color analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorAnalysis {
    /// Dominant colors (RGB)
    pub dominant_colors: Vec<DominantColor>,
    /// Average saturation (0.0-1.0)
    pub avg_saturation: f64,
    /// Color diversity (0.0-1.0)
    pub color_diversity: f64,
    /// Detected color grading style
    pub grading_style: ColorGradingStyle,
}

/// Dominant color information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominantColor {
    /// RGB color (0-255 each)
    pub rgb: (u8, u8, u8),
    /// Percentage of pixels (0.0-1.0)
    pub percentage: f64,
}

/// Color grading style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorGradingStyle {
    /// Natural/neutral colors
    Natural,
    /// Warm tones (orange/yellow bias)
    Warm,
    /// Cool tones (blue bias)
    Cool,
    /// High contrast
    HighContrast,
    /// Desaturated/bleach bypass
    Desaturated,
    /// Vibrant/saturated
    Vibrant,
}

/// Color analyzer.
pub struct ColorAnalyzer {
    target_colors: usize,
    color_samples: Vec<YuvPixel>,
    saturation_sum: f64,
    sample_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct YuvPixel {
    y: u8,
    u: u8,
    v: u8,
}

impl ColorAnalyzer {
    /// Create a new color analyzer.
    ///
    /// # Parameters
    ///
    /// - `target_colors`: Number of dominant colors to extract
    #[must_use]
    pub fn new(target_colors: usize) -> Self {
        Self {
            target_colors,
            color_samples: Vec::new(),
            saturation_sum: 0.0,
            sample_count: 0,
        }
    }

    /// Process a frame.
    pub fn process_frame(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        width: usize,
        height: usize,
        _frame_number: usize,
    ) -> AnalysisResult<()> {
        if y_plane.len() != width * height {
            return Err(AnalysisError::InvalidInput(
                "Y plane size mismatch".to_string(),
            ));
        }

        // For YUV420p, U and V planes are half resolution
        let uv_width = width.div_ceil(2);
        let uv_height = height.div_ceil(2);
        if u_plane.len() != uv_width * uv_height || v_plane.len() != uv_width * uv_height {
            return Err(AnalysisError::InvalidInput(
                "UV plane size mismatch".to_string(),
            ));
        }

        // Sample pixels (not every pixel to avoid memory issues)
        const SAMPLE_STEP: usize = 8;

        for y in (0..height).step_by(SAMPLE_STEP) {
            for x in (0..width).step_by(SAMPLE_STEP) {
                let y_val = y_plane[y * width + x];
                let uv_x = x / 2;
                let uv_y = y / 2;
                let u_val = u_plane[uv_y * uv_width + uv_x];
                let v_val = v_plane[uv_y * uv_width + uv_x];

                self.color_samples.push(YuvPixel {
                    y: y_val,
                    u: u_val,
                    v: v_val,
                });

                // Compute saturation
                let saturation = compute_saturation(u_val, v_val);
                self.saturation_sum += saturation;
                self.sample_count += 1;
            }
        }

        Ok(())
    }

    /// Finalize and return color analysis.
    pub fn finalize(self) -> ColorAnalysis {
        if self.color_samples.is_empty() {
            return ColorAnalysis {
                dominant_colors: Vec::new(),
                avg_saturation: 0.0,
                color_diversity: 0.0,
                grading_style: ColorGradingStyle::Natural,
            };
        }

        // Extract dominant colors using k-means
        let dominant_colors = extract_dominant_colors(&self.color_samples, self.target_colors);

        // Compute average saturation
        let avg_saturation = if self.sample_count > 0 {
            self.saturation_sum / self.sample_count as f64
        } else {
            0.0
        };

        // Compute color diversity (spread of colors)
        let color_diversity = compute_color_diversity(&dominant_colors);

        // Determine grading style
        let grading_style = determine_grading_style(&dominant_colors, avg_saturation);

        ColorAnalysis {
            dominant_colors,
            avg_saturation,
            color_diversity,
            grading_style,
        }
    }
}

/// Convert YUV to RGB.
fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y = f64::from(y);
    let u = f64::from(u) - 128.0;
    let v = f64::from(v) - 128.0;

    let r = y + 1.402 * v;
    let g = y - 0.344136 * u - 0.714136 * v;
    let b = y + 1.772 * u;

    let r = r.clamp(0.0, 255.0) as u8;
    let g = g.clamp(0.0, 255.0) as u8;
    let b = b.clamp(0.0, 255.0) as u8;

    (r, g, b)
}

/// Compute saturation from UV values.
fn compute_saturation(u: u8, v: u8) -> f64 {
    let u = f64::from(u) - 128.0;
    let v = f64::from(v) - 128.0;
    let chroma = (u * u + v * v).sqrt();
    (chroma / 181.0).min(1.0) // Max chroma ≈ 181
}

/// Extract dominant colors using simplified k-means.
fn extract_dominant_colors(samples: &[YuvPixel], k: usize) -> Vec<DominantColor> {
    if samples.is_empty() || k == 0 {
        return Vec::new();
    }

    let k = k.min(samples.len());

    // Initialize centroids with random samples
    let mut centroids: Vec<YuvPixel> = Vec::with_capacity(k);
    let step = samples.len() / k;
    for i in 0..k {
        centroids.push(samples[i * step]);
    }

    // Run k-means iterations
    const MAX_ITERATIONS: usize = 10;
    for _ in 0..MAX_ITERATIONS {
        // Assign each sample to nearest centroid
        let mut clusters: Vec<Vec<YuvPixel>> = vec![Vec::new(); k];

        for &sample in samples {
            let mut min_dist = f64::MAX;
            let mut best_cluster = 0;

            for (i, &centroid) in centroids.iter().enumerate() {
                let dist = color_distance(sample, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = i;
                }
            }

            clusters[best_cluster].push(sample);
        }

        // Update centroids
        for (i, cluster) in clusters.iter().enumerate() {
            if !cluster.is_empty() {
                let avg_y = cluster.iter().map(|p| p.y as usize).sum::<usize>() / cluster.len();
                let avg_u = cluster.iter().map(|p| p.u as usize).sum::<usize>() / cluster.len();
                let avg_v = cluster.iter().map(|p| p.v as usize).sum::<usize>() / cluster.len();

                centroids[i] = YuvPixel {
                    y: avg_y as u8,
                    u: avg_u as u8,
                    v: avg_v as u8,
                };
            }
        }
    }

    // Convert centroids to RGB and compute percentages
    let mut dominant_colors: Vec<_> = centroids
        .into_iter()
        .map(|yuv| {
            let rgb = yuv_to_rgb(yuv.y, yuv.u, yuv.v);
            DominantColor {
                rgb,
                percentage: 0.0,
            }
        })
        .collect();

    // Compute actual percentages
    for color in &mut dominant_colors {
        let count = samples
            .iter()
            .filter(|&&sample| {
                let rgb = yuv_to_rgb(sample.y, sample.u, sample.v);
                rgb_distance(rgb, color.rgb) < 50.0
            })
            .count();
        color.percentage = count as f64 / samples.len() as f64;
    }

    // Sort by percentage
    dominant_colors.sort_by(|a, b| {
        b.percentage
            .partial_cmp(&a.percentage)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    dominant_colors
}

/// Compute distance between two YUV colors.
fn color_distance(a: YuvPixel, b: YuvPixel) -> f64 {
    let dy = f64::from(a.y) - f64::from(b.y);
    let du = f64::from(a.u) - f64::from(b.u);
    let dv = f64::from(a.v) - f64::from(b.v);
    (dy * dy + du * du + dv * dv).sqrt()
}

/// Compute distance between two RGB colors.
fn rgb_distance(a: (u8, u8, u8), b: (u8, u8, u8)) -> f64 {
    let dr = f64::from(a.0) - f64::from(b.0);
    let dg = f64::from(a.1) - f64::from(b.1);
    let db = f64::from(a.2) - f64::from(b.2);
    (dr * dr + dg * dg + db * db).sqrt()
}

/// Compute color diversity.
fn compute_color_diversity(colors: &[DominantColor]) -> f64 {
    if colors.len() < 2 {
        return 0.0;
    }

    // Measure how evenly distributed the colors are
    let entropy: f64 = colors
        .iter()
        .filter(|c| c.percentage > 0.0)
        .map(|c| {
            let p = c.percentage;
            -p * p.log2()
        })
        .sum();

    let max_entropy = (colors.len() as f64).log2();
    if max_entropy > 0.0 {
        (entropy / max_entropy).min(1.0)
    } else {
        0.0
    }
}

/// Determine color grading style.
fn determine_grading_style(colors: &[DominantColor], avg_saturation: f64) -> ColorGradingStyle {
    if colors.is_empty() {
        return ColorGradingStyle::Natural;
    }

    // Check saturation
    if avg_saturation < 0.2 {
        return ColorGradingStyle::Desaturated;
    }
    if avg_saturation > 0.7 {
        return ColorGradingStyle::Vibrant;
    }

    // Check color temperature
    let mut warm_score = 0.0;
    let mut cool_score = 0.0;

    for color in colors {
        let (r, g, b) = color.rgb;
        // Warm: red/orange dominant
        if r > g && r > b {
            warm_score += color.percentage;
        }
        // Cool: blue dominant
        if b > r && b > g {
            cool_score += color.percentage;
        }
    }

    if warm_score > cool_score * 1.5 {
        ColorGradingStyle::Warm
    } else if cool_score > warm_score * 1.5 {
        ColorGradingStyle::Cool
    } else {
        ColorGradingStyle::Natural
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuv_to_rgb() {
        // Black
        let (r, g, b) = yuv_to_rgb(16, 128, 128);
        assert!(r < 20 && g < 20 && b < 20);

        // White
        let (r, g, b) = yuv_to_rgb(235, 128, 128);
        assert!(r > 230 && g > 230 && b > 230);
    }

    #[test]
    fn test_saturation() {
        // Gray (no saturation)
        let sat = compute_saturation(128, 128);
        assert!(sat < 0.1);

        // Saturated color
        let sat = compute_saturation(200, 200);
        assert!(sat > 0.3);
    }

    #[test]
    fn test_color_analyzer() {
        let mut analyzer = ColorAnalyzer::new(5);

        // Process a frame
        let y_plane = vec![128u8; 64 * 64];
        let u_plane = vec![128u8; 32 * 32];
        let v_plane = vec![128u8; 32 * 32];

        analyzer
            .process_frame(&y_plane, &u_plane, &v_plane, 64, 64, 0)
            .expect("unexpected None/Err");

        let analysis = analyzer.finalize();
        assert!(!analysis.dominant_colors.is_empty());
    }

    #[test]
    fn test_dominant_colors() {
        // Create samples with two distinct colors
        let mut samples = Vec::new();
        for _ in 0..100 {
            samples.push(YuvPixel {
                y: 100,
                u: 128,
                v: 128,
            });
        }
        for _ in 0..100 {
            samples.push(YuvPixel {
                y: 200,
                u: 128,
                v: 128,
            });
        }

        let colors = extract_dominant_colors(&samples, 2);
        assert_eq!(colors.len(), 2);
    }

    #[test]
    fn test_color_diversity() {
        // Even distribution
        let colors = vec![
            DominantColor {
                rgb: (255, 0, 0),
                percentage: 0.25,
            },
            DominantColor {
                rgb: (0, 255, 0),
                percentage: 0.25,
            },
            DominantColor {
                rgb: (0, 0, 255),
                percentage: 0.25,
            },
            DominantColor {
                rgb: (255, 255, 0),
                percentage: 0.25,
            },
        ];
        let diversity = compute_color_diversity(&colors);
        assert!(diversity > 0.9);

        // Uneven distribution
        let colors2 = vec![
            DominantColor {
                rgb: (255, 0, 0),
                percentage: 0.9,
            },
            DominantColor {
                rgb: (0, 255, 0),
                percentage: 0.1,
            },
        ];
        let diversity2 = compute_color_diversity(&colors2);
        assert!(diversity2 < 0.8);
    }

    #[test]
    fn test_grading_detection() {
        // Desaturated
        let colors = vec![DominantColor {
            rgb: (128, 128, 128),
            percentage: 1.0,
        }];
        let style = determine_grading_style(&colors, 0.1);
        assert_eq!(style, ColorGradingStyle::Desaturated);

        // Vibrant
        let style2 = determine_grading_style(&colors, 0.8);
        assert_eq!(style2, ColorGradingStyle::Vibrant);
    }

    #[test]
    fn test_empty_analyzer() {
        let analyzer = ColorAnalyzer::new(5);
        let analysis = analyzer.finalize();
        assert!(analysis.dominant_colors.is_empty());
        assert_eq!(analysis.grading_style, ColorGradingStyle::Natural);
    }
}
