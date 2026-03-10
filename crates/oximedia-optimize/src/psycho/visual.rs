//! Psychovisual optimization implementation.

use crate::OptimizerConfig;
use oximedia_core::OxiResult;

/// Psychovisual analyzer for perceptual quality optimization.
pub struct PsychoAnalyzer {
    enable_edge_preservation: bool,
    enable_texture_optimization: bool,
    masking_strength: f64,
}

impl PsychoAnalyzer {
    /// Creates a new psychovisual analyzer.
    pub fn new(config: &OptimizerConfig) -> OxiResult<Self> {
        Ok(Self {
            enable_edge_preservation: config.enable_psychovisual,
            enable_texture_optimization: config.enable_psychovisual,
            masking_strength: 1.0,
        })
    }

    /// Analyzes edge content in a block.
    #[must_use]
    pub fn analyze_edges(&self, pixels: &[u8], width: usize) -> EdgeAnalysis {
        if !self.enable_edge_preservation || pixels.is_empty() {
            return EdgeAnalysis::default();
        }

        let height = pixels.len() / width;
        let mut edge_strength = 0.0;
        let mut edge_count = 0;

        // Sobel edge detection
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                let gx = sobel_x(pixels, idx, width);
                let gy = sobel_y(pixels, idx, width);
                let magnitude = f64::from(gx * gx + gy * gy).sqrt();

                if magnitude > 10.0 {
                    edge_strength += magnitude;
                    edge_count += 1;
                }
            }
        }

        EdgeAnalysis {
            edge_strength: if edge_count > 0 {
                edge_strength / edge_count as f64
            } else {
                0.0
            },
            edge_density: edge_count as f64 / ((width * height) as f64),
            preserve_edges: edge_count > (width * height / 10),
        }
    }

    /// Analyzes texture content in a block.
    #[must_use]
    pub fn analyze_texture(&self, pixels: &[u8]) -> TextureAnalysis {
        if !self.enable_texture_optimization || pixels.is_empty() {
            return TextureAnalysis::default();
        }

        // Calculate variance as texture metric
        let mean = pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / pixels.len() as f64;
        let variance = pixels
            .iter()
            .map(|&p| {
                let diff = f64::from(p) - mean;
                diff * diff
            })
            .sum::<f64>()
            / pixels.len() as f64;

        TextureAnalysis {
            variance,
            is_textured: variance > 100.0,
            complexity: (variance / 255.0).min(1.0),
        }
    }

    /// Calculates psychovisual weight for quantization.
    #[must_use]
    pub fn calculate_psycho_weight(&self, edge: &EdgeAnalysis, texture: &TextureAnalysis) -> f64 {
        let mut weight = 1.0;

        // Preserve edges by reducing quantization
        if edge.preserve_edges {
            weight *= 0.8 - 0.2 * edge.edge_density;
        }

        // Allow more aggressive quantization in textured areas
        if texture.is_textured {
            weight *= 1.0 + 0.3 * texture.complexity;
        }

        weight * self.masking_strength
    }
}

/// Sobel operator for horizontal gradient.
fn sobel_x(pixels: &[u8], idx: usize, width: usize) -> i32 {
    let p = |dy: isize, dx: isize| {
        let i = (idx as isize + dy * width as isize + dx) as usize;
        i32::from(pixels[i])
    };

    -p(-1, -1) - 2 * p(0, -1) - p(1, -1) + p(-1, 1) + 2 * p(0, 1) + p(1, 1)
}

/// Sobel operator for vertical gradient.
fn sobel_y(pixels: &[u8], idx: usize, width: usize) -> i32 {
    let p = |dy: isize, dx: isize| {
        let i = (idx as isize + dy * width as isize + dx) as usize;
        i32::from(pixels[i])
    };

    -p(-1, -1) - 2 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2 * p(1, 0) + p(1, 1)
}

/// Edge analysis result.
#[derive(Debug, Clone, Copy)]
pub struct EdgeAnalysis {
    /// Average edge strength.
    pub edge_strength: f64,
    /// Proportion of edge pixels.
    pub edge_density: f64,
    /// Whether to preserve edges.
    pub preserve_edges: bool,
}

impl Default for EdgeAnalysis {
    fn default() -> Self {
        Self {
            edge_strength: 0.0,
            edge_density: 0.0,
            preserve_edges: false,
        }
    }
}

/// Texture analysis result.
#[derive(Debug, Clone, Copy)]
pub struct TextureAnalysis {
    /// Pixel variance.
    pub variance: f64,
    /// Whether the block is textured.
    pub is_textured: bool,
    /// Texture complexity (0-1).
    pub complexity: f64,
}

impl Default for TextureAnalysis {
    fn default() -> Self {
        Self {
            variance: 0.0,
            is_textured: false,
            complexity: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psycho_analyzer_creation() {
        let config = OptimizerConfig::default();
        let analyzer =
            PsychoAnalyzer::new(&config).expect("psycho analyzer creation should succeed");
        assert!(analyzer.enable_edge_preservation);
    }

    #[test]
    fn test_edge_analysis_flat() {
        let config = OptimizerConfig::default();
        let analyzer =
            PsychoAnalyzer::new(&config).expect("psycho analyzer creation should succeed");
        let pixels = vec![128u8; 64]; // Flat block
        let analysis = analyzer.analyze_edges(&pixels, 8);
        assert!(!analysis.preserve_edges);
        assert_eq!(analysis.edge_density, 0.0);
    }

    #[test]
    fn test_texture_analysis_flat() {
        let config = OptimizerConfig::default();
        let analyzer =
            PsychoAnalyzer::new(&config).expect("psycho analyzer creation should succeed");
        let pixels = vec![128u8; 64]; // Flat block
        let analysis = analyzer.analyze_texture(&pixels);
        assert!(!analysis.is_textured);
        assert_eq!(analysis.variance, 0.0);
    }

    #[test]
    fn test_texture_analysis_varied() {
        let config = OptimizerConfig::default();
        let analyzer =
            PsychoAnalyzer::new(&config).expect("psycho analyzer creation should succeed");
        let mut pixels = vec![0u8; 64];
        for (i, pixel) in pixels.iter_mut().enumerate() {
            *pixel = (i * 4) as u8;
        }
        let analysis = analyzer.analyze_texture(&pixels);
        assert!(analysis.variance > 0.0);
    }
}
