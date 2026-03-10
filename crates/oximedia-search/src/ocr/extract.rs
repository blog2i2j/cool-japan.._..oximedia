//! OCR text extraction.
//!
//! Provides a pure-Rust connected-component analysis approach that
//! identifies text-like regions in an image based on geometric
//! properties (aspect ratio, fill ratio, component size).  This is a
//! structural text detector; for full character recognition an
//! external model would be needed.  The detector outputs bounding
//! boxes for text-like regions and a placeholder string indicating
//! detected region count.

use crate::error::SearchResult;

/// Minimum component area (in pixels) to consider as potential text.
const MIN_COMPONENT_AREA: usize = 8;
/// Maximum aspect ratio (width / height) for a text-like component.
const MAX_ASPECT_RATIO: f32 = 10.0;
/// Minimum fill ratio (component pixels / bounding-box area).
const MIN_FILL_RATIO: f32 = 0.15;

/// Bounding box of a detected text region.
#[derive(Debug, Clone)]
pub struct TextRegion {
    /// X coordinate of the top-left corner.
    pub x: u32,
    /// Y coordinate of the top-left corner.
    pub y: u32,
    /// Width of the region.
    pub width: u32,
    /// Height of the region.
    pub height: u32,
    /// Number of foreground pixels in the region.
    pub pixel_count: u32,
}

/// OCR text extractor using connected-component analysis.
pub struct OcrExtractor {
    /// Binarisation threshold (0-255).
    threshold: u8,
}

impl OcrExtractor {
    /// Create a new OCR extractor with default threshold.
    #[must_use]
    pub const fn new() -> Self {
        Self { threshold: 128 }
    }

    /// Create an extractor with a custom binarisation threshold.
    #[must_use]
    pub const fn with_threshold(threshold: u8) -> Self {
        Self { threshold }
    }

    /// Detect text-like regions in raw RGB image data.
    ///
    /// Returns bounding boxes for regions that pass geometric filters.
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_regions(&self, image_data: &[u8]) -> SearchResult<Vec<TextRegion>> {
        if image_data.len() < 3 {
            return Ok(Vec::new());
        }

        let pixel_count = image_data.len() / 3;
        let width = (pixel_count as f64).sqrt() as usize;
        if width < 4 {
            return Ok(Vec::new());
        }
        let height = pixel_count / width;

        // Convert to grayscale and binarise.
        let binary: Vec<bool> = image_data
            .chunks_exact(3)
            .take(width * height)
            .map(|rgb| {
                let gray = (0.299 * f32::from(rgb[0])
                    + 0.587 * f32::from(rgb[1])
                    + 0.114 * f32::from(rgb[2])) as u8;
                gray < self.threshold
            })
            .collect();

        // Simple connected-component labelling (4-connectivity).
        let mut labels = vec![0u32; width * height];
        let mut next_label = 1u32;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if !binary[idx] {
                    continue;
                }
                // Check left and top neighbours.
                let left = if x > 0 { labels[idx - 1] } else { 0 };
                let top = if y > 0 { labels[idx - width] } else { 0 };

                if left == 0 && top == 0 {
                    labels[idx] = next_label;
                    next_label = next_label.saturating_add(1);
                } else if left != 0 && top == 0 {
                    labels[idx] = left;
                } else if left == 0 && top != 0 {
                    labels[idx] = top;
                } else {
                    // Both neighbours labelled; use the smaller label.
                    let min_label = left.min(top);
                    let max_label = left.max(top);
                    labels[idx] = min_label;
                    // Union: relabel max_label to min_label.
                    if min_label != max_label {
                        for l in labels.iter_mut() {
                            if *l == max_label {
                                *l = min_label;
                            }
                        }
                    }
                }
            }
        }

        // Collect bounding boxes per label.
        let mut components: std::collections::HashMap<u32, (usize, usize, usize, usize, u32)> =
            std::collections::HashMap::new();

        for y in 0..height {
            for x in 0..width {
                let label = labels[y * width + x];
                if label == 0 {
                    continue;
                }
                let entry = components.entry(label).or_insert((x, y, x, y, 0));
                entry.0 = entry.0.min(x);
                entry.1 = entry.1.min(y);
                entry.2 = entry.2.max(x);
                entry.3 = entry.3.max(y);
                entry.4 += 1;
            }
        }

        // Filter by geometric properties.
        let regions: Vec<TextRegion> = components
            .values()
            .filter_map(|&(x0, y0, x1, y1, count)| {
                let w = (x1 - x0 + 1) as u32;
                let h = (y1 - y0 + 1) as u32;
                let area = (w * h) as usize;
                let component_area = count as usize;

                if component_area < MIN_COMPONENT_AREA {
                    return None;
                }

                let aspect = w as f32 / h.max(1) as f32;
                if aspect > MAX_ASPECT_RATIO {
                    return None;
                }

                let fill = component_area as f32 / area.max(1) as f32;
                if fill < MIN_FILL_RATIO {
                    return None;
                }

                Some(TextRegion {
                    x: x0 as u32,
                    y: y0 as u32,
                    width: w,
                    height: h,
                    pixel_count: count,
                })
            })
            .collect();

        Ok(regions)
    }

    /// Extract text from image data.
    ///
    /// Currently performs structural text region detection and returns
    /// a summary string describing found regions. Full character
    /// recognition would require a neural network model.
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails.
    pub fn extract(&self, image_data: &[u8]) -> SearchResult<String> {
        let regions = self.detect_regions(image_data)?;
        if regions.is_empty() {
            return Ok(String::new());
        }

        let mut result = format!("[{} text region(s) detected]", regions.len());
        for (i, region) in regions.iter().enumerate() {
            result.push_str(&format!(
                " Region {}: {}x{} at ({},{})",
                i + 1,
                region.width,
                region.height,
                region.x,
                region.y
            ));
        }

        Ok(result)
    }
}

impl Default for OcrExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_empty_image() {
        let extractor = OcrExtractor::new();
        let result = extractor.extract(&[]).expect("should succeed in test");
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_uniform_image_no_text() {
        let extractor = OcrExtractor::new();
        // Bright uniform image => no foreground after binarisation.
        let data = vec![200u8; 16 * 16 * 3];
        let result = extractor.extract(&data).expect("should succeed in test");
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_regions_with_dark_block() {
        let extractor = OcrExtractor::new();
        // Create a 16x16 white image with a 4x4 dark block.
        let mut data = vec![200u8; 16 * 16 * 3];
        for y in 4..8 {
            for x in 4..8 {
                let idx = (y * 16 + x) * 3;
                data[idx] = 20;
                data[idx + 1] = 20;
                data[idx + 2] = 20;
            }
        }
        let regions = extractor
            .detect_regions(&data)
            .expect("should succeed in test");
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_custom_threshold() {
        let extractor = OcrExtractor::with_threshold(50);
        // Only very dark pixels are foreground.
        let data = vec![100u8; 16 * 16 * 3];
        let regions = extractor
            .detect_regions(&data)
            .expect("should succeed in test");
        // All pixels are > 50 grayscale, so no foreground.
        assert!(regions.is_empty());
    }
}
