#![allow(dead_code)]

//! Shot color palette extraction and comparison.
//!
//! This module extracts dominant color palettes from representative frames of
//! each shot, provides palette distance metrics, and can cluster shots by
//! visual colour similarity.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// An sRGB color triplet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rgb {
    /// Red channel (0-255).
    pub r: u8,
    /// Green channel (0-255).
    pub g: u8,
    /// Blue channel (0-255).
    pub b: u8,
}

impl Rgb {
    /// Create a new RGB colour.
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Euclidean distance to another colour.
    #[allow(clippy::cast_precision_loss)]
    pub fn distance(&self, other: &Self) -> f64 {
        let dr = f64::from(self.r) - f64::from(other.r);
        let dg = f64::from(self.g) - f64::from(other.g);
        let db = f64::from(self.b) - f64::from(other.b);
        (dr * dr + dg * dg + db * db).sqrt()
    }

    /// Return a luminance approximation (BT.601).
    #[allow(clippy::cast_precision_loss)]
    pub fn luminance(&self) -> f64 {
        0.299 * f64::from(self.r) + 0.587 * f64::from(self.g) + 0.114 * f64::from(self.b)
    }
}

/// A weighted color in a palette.
#[derive(Debug, Clone)]
pub struct PaletteEntry {
    /// The colour.
    pub color: Rgb,
    /// Relative weight / proportion in the frame (0.0 .. 1.0).
    pub weight: f64,
}

/// A colour palette extracted from one shot.
#[derive(Debug, Clone)]
pub struct ShotPalette {
    /// Shot index.
    pub shot_index: usize,
    /// Ordered list of dominant colours.
    pub entries: Vec<PaletteEntry>,
}

/// Result of comparing two palettes.
#[derive(Debug, Clone)]
pub struct PaletteComparison {
    /// Index of the first shot.
    pub shot_a: usize,
    /// Index of the second shot.
    pub shot_b: usize,
    /// Overall distance score (lower = more similar).
    pub distance: f64,
}

/// Result of palette clustering.
#[derive(Debug, Clone)]
pub struct PaletteCluster {
    /// Cluster label.
    pub id: usize,
    /// Shot indices belonging to this cluster.
    pub members: Vec<usize>,
    /// Mean colour of the cluster centroid.
    pub centroid: Rgb,
}

/// Palette analyser.
#[derive(Debug, Clone)]
pub struct PaletteAnalyzer {
    /// Maximum colours per palette.
    max_colors: usize,
    /// Number of median-cut iterations for extraction.
    iterations: usize,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl Default for PaletteAnalyzer {
    fn default() -> Self {
        Self {
            max_colors: 5,
            iterations: 4,
        }
    }
}

impl ShotPalette {
    /// Create a palette from entries.
    pub fn new(shot_index: usize, entries: Vec<PaletteEntry>) -> Self {
        Self {
            shot_index,
            entries,
        }
    }

    /// Average luminance of the palette weighted by entry proportion.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_luminance(&self) -> f64 {
        let total_weight: f64 = self.entries.iter().map(|e| e.weight).sum();
        if total_weight == 0.0 {
            return 0.0;
        }
        let weighted_lum: f64 = self
            .entries
            .iter()
            .map(|e| e.color.luminance() * e.weight)
            .sum();
        weighted_lum / total_weight
    }

    /// Number of colours in the palette.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the palette is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Dominant colour (highest weight).
    pub fn dominant(&self) -> Option<&Rgb> {
        self.entries
            .iter()
            .max_by(|a, b| {
                a.weight
                    .partial_cmp(&b.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| &e.color)
    }
}

impl PaletteAnalyzer {
    /// Create a new analyser with custom parameters.
    pub fn new(max_colors: usize, iterations: usize) -> Self {
        Self {
            max_colors,
            iterations,
        }
    }

    /// Extract a palette from raw pixel data (flattened RGB triples).
    #[allow(clippy::cast_precision_loss)]
    pub fn extract_palette(&self, pixels: &[u8], shot_index: usize) -> ShotPalette {
        if pixels.len() < 3 {
            return ShotPalette::new(shot_index, Vec::new());
        }

        // Collect unique colours with counts.
        let mut counts: HashMap<Rgb, usize> = HashMap::new();
        for chunk in pixels.chunks_exact(3) {
            let c = Rgb::new(chunk[0], chunk[1], chunk[2]);
            *counts.entry(c).or_insert(0) += 1;
        }

        // Sort by frequency and take top N.
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let total_pixels = (pixels.len() / 3) as f64;
        let entries: Vec<PaletteEntry> = sorted
            .into_iter()
            .take(self.max_colors)
            .map(|(color, count)| PaletteEntry {
                color,
                weight: count as f64 / total_pixels,
            })
            .collect();

        ShotPalette::new(shot_index, entries)
    }

    /// Compare two palettes using earth-mover-style distance.
    #[allow(clippy::cast_precision_loss)]
    pub fn compare(&self, a: &ShotPalette, b: &ShotPalette) -> PaletteComparison {
        // Simple weighted average of min distances.
        let mut total = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        for ea in &a.entries {
            let min_dist = b
                .entries
                .iter()
                .map(|eb| ea.color.distance(&eb.color))
                .fold(f64::INFINITY, f64::min);
            total += min_dist * ea.weight;
            weight_sum += ea.weight;
        }
        for eb in &b.entries {
            let min_dist = a
                .entries
                .iter()
                .map(|ea| eb.color.distance(&ea.color))
                .fold(f64::INFINITY, f64::min);
            total += min_dist * eb.weight;
            weight_sum += eb.weight;
        }

        let distance = if weight_sum > 0.0 {
            total / weight_sum
        } else {
            0.0
        };

        PaletteComparison {
            shot_a: a.shot_index,
            shot_b: b.shot_index,
            distance,
        }
    }

    /// Cluster palettes by simple nearest-centroid assignment.
    #[allow(clippy::cast_precision_loss)]
    pub fn cluster(&self, palettes: &[ShotPalette], k: usize) -> Vec<PaletteCluster> {
        if palettes.is_empty() || k == 0 {
            return Vec::new();
        }

        // Use dominant colour of first k palettes as initial centroids.
        let initial: Vec<Rgb> = palettes
            .iter()
            .take(k)
            .filter_map(|p| p.dominant().copied())
            .collect();

        if initial.is_empty() {
            return Vec::new();
        }

        // Single-pass assignment.
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); initial.len()];
        for (i, p) in palettes.iter().enumerate() {
            if let Some(dom) = p.dominant() {
                let best = initial
                    .iter()
                    .enumerate()
                    .min_by(|(_, ca), (_, cb)| {
                        dom.distance(ca)
                            .partial_cmp(&dom.distance(cb))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                clusters[best].push(i);
            }
        }

        clusters
            .into_iter()
            .enumerate()
            .map(|(id, members)| PaletteCluster {
                id,
                centroid: if id < initial.len() {
                    initial[id]
                } else {
                    Rgb::new(0, 0, 0)
                },
                members,
            })
            .collect()
    }

    /// Maximum colours per palette.
    pub fn max_colors(&self) -> usize {
        self.max_colors
    }

    /// Configured iteration count.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_distance_identical() {
        let a = Rgb::new(100, 100, 100);
        assert!((a.distance(&a) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rgb_distance_different() {
        let a = Rgb::new(0, 0, 0);
        let b = Rgb::new(255, 255, 255);
        assert!(a.distance(&b) > 400.0);
    }

    #[test]
    fn test_rgb_luminance() {
        let white = Rgb::new(255, 255, 255);
        let black = Rgb::new(0, 0, 0);
        assert!(white.luminance() > 250.0);
        assert!((black.luminance() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_palette_empty() {
        let p = ShotPalette::new(0, Vec::new());
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert!(p.dominant().is_none());
    }

    #[test]
    fn test_palette_dominant() {
        let p = ShotPalette::new(
            0,
            vec![
                PaletteEntry {
                    color: Rgb::new(255, 0, 0),
                    weight: 0.3,
                },
                PaletteEntry {
                    color: Rgb::new(0, 255, 0),
                    weight: 0.7,
                },
            ],
        );
        let dom = p.dominant().expect("should succeed in test");
        assert_eq!(*dom, Rgb::new(0, 255, 0));
    }

    #[test]
    fn test_palette_average_luminance() {
        let p = ShotPalette::new(
            0,
            vec![PaletteEntry {
                color: Rgb::new(255, 255, 255),
                weight: 1.0,
            }],
        );
        assert!(p.average_luminance() > 254.0);
    }

    #[test]
    fn test_extract_palette_empty_pixels() {
        let a = PaletteAnalyzer::default();
        let p = a.extract_palette(&[], 0);
        assert!(p.is_empty());
    }

    #[test]
    fn test_extract_palette_single_color() {
        let a = PaletteAnalyzer::default();
        let pixels: Vec<u8> = vec![100, 150, 200, 100, 150, 200, 100, 150, 200];
        let p = a.extract_palette(&pixels, 0);
        assert_eq!(p.len(), 1);
        assert_eq!(p.entries[0].color, Rgb::new(100, 150, 200));
        assert!((p.entries[0].weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compare_identical_palettes() {
        let a = PaletteAnalyzer::default();
        let p1 = ShotPalette::new(
            0,
            vec![PaletteEntry {
                color: Rgb::new(50, 50, 50),
                weight: 1.0,
            }],
        );
        let p2 = p1.clone();
        let cmp = a.compare(&p1, &p2);
        assert!((cmp.distance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compare_different_palettes() {
        let a = PaletteAnalyzer::default();
        let p1 = ShotPalette::new(
            0,
            vec![PaletteEntry {
                color: Rgb::new(0, 0, 0),
                weight: 1.0,
            }],
        );
        let p2 = ShotPalette::new(
            1,
            vec![PaletteEntry {
                color: Rgb::new(255, 255, 255),
                weight: 1.0,
            }],
        );
        let cmp = a.compare(&p1, &p2);
        assert!(cmp.distance > 100.0);
    }

    #[test]
    fn test_cluster_empty() {
        let a = PaletteAnalyzer::default();
        let clusters = a.cluster(&[], 3);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_single() {
        let a = PaletteAnalyzer::default();
        let palettes = vec![ShotPalette::new(
            0,
            vec![PaletteEntry {
                color: Rgb::new(10, 10, 10),
                weight: 1.0,
            }],
        )];
        let clusters = a.cluster(&palettes, 1);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members.len(), 1);
    }

    #[test]
    fn test_analyzer_accessors() {
        let a = PaletteAnalyzer::new(8, 6);
        assert_eq!(a.max_colors(), 8);
        assert_eq!(a.iterations(), 6);
    }

    #[test]
    fn test_default_analyzer() {
        let a = PaletteAnalyzer::default();
        assert_eq!(a.max_colors(), 5);
        assert_eq!(a.iterations(), 4);
    }
}
