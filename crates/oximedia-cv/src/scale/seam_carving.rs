//! Seam carving implementation for content-aware image resizing.
//!
//! Seam carving is a content-aware image resizing technique that removes
//! or inserts seams (connected paths of pixels) with minimal energy.
//! This allows for intelligent resizing that preserves important image features.

use super::energy::{compute_cumulative_energy, EnergyFunction, EnergyMap};
use crate::error::{CvError, CvResult};

/// A seam through an image.
///
/// A seam is a connected path of pixels from one edge to another.
/// For vertical seams, the path goes from top to bottom.
/// For horizontal seams, the path goes from left to right.
#[derive(Debug, Clone)]
pub struct Seam {
    /// Pixel coordinates in the seam (x for vertical, y for horizontal).
    pub path: Vec<u32>,
    /// Total energy of the seam.
    pub energy: f64,
    /// Whether this is a vertical seam.
    pub vertical: bool,
}

impl Seam {
    /// Create a new seam.
    #[must_use]
    pub fn new(path: Vec<u32>, energy: f64, vertical: bool) -> Self {
        Self {
            path,
            energy,
            vertical,
        }
    }

    /// Get the length of the seam.
    #[must_use]
    pub fn len(&self) -> usize {
        self.path.len()
    }

    /// Check if seam is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }
}

/// Seam carver for content-aware image resizing.
#[derive(Debug, Clone)]
pub struct SeamCarver {
    /// Energy function to use.
    energy_function: EnergyFunction,
    /// Protection mask (optional).
    protection_mask: Option<Vec<u8>>,
    /// Protection energy scale factor.
    protection_scale: f64,
}

impl SeamCarver {
    /// Create a new seam carver with the given energy function.
    #[must_use]
    pub fn new(energy_function: EnergyFunction) -> Self {
        Self {
            energy_function,
            protection_mask: None,
            protection_scale: 1000.0,
        }
    }

    /// Set protection mask.
    ///
    /// Protected regions (mask value > 0) will have increased energy
    /// to prevent them from being removed.
    pub fn set_protection_mask(&mut self, mask: Vec<u8>) {
        self.protection_mask = Some(mask);
    }

    /// Set protection energy scale.
    pub fn set_protection_scale(&mut self, scale: f64) {
        self.protection_scale = scale;
    }

    /// Find the optimal vertical seam in an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// The lowest-energy vertical seam.
    pub fn find_vertical_seam(&self, image: &[u8], width: u32, height: u32) -> CvResult<Seam> {
        let energy = self.compute_energy(image, width, height)?;
        Ok(find_min_vertical_seam(&energy))
    }

    /// Find the optimal horizontal seam in an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// The lowest-energy horizontal seam.
    pub fn find_horizontal_seam(&self, image: &[u8], width: u32, height: u32) -> CvResult<Seam> {
        let energy = self.compute_energy(image, width, height)?;
        Ok(find_min_horizontal_seam(&energy))
    }

    /// Remove a vertical seam from a grayscale image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `seam` - Seam to remove
    ///
    /// # Returns
    ///
    /// Image with seam removed (width reduced by 1).
    pub fn remove_vertical_seam(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        seam: &Seam,
    ) -> CvResult<Vec<u8>> {
        if !seam.vertical {
            return Err(CvError::invalid_parameter("seam", "expected vertical seam"));
        }

        if seam.path.len() != height as usize {
            return Err(CvError::invalid_parameter(
                "seam.path.len()",
                format!("expected {}, got {}", height, seam.path.len()),
            ));
        }

        let new_width = width - 1;
        let mut result = vec![0u8; new_width as usize * height as usize];

        for y in 0..height as usize {
            let seam_x = seam.path[y] as usize;
            let src_row_start = y * width as usize;
            let dst_row_start = y * new_width as usize;

            // Copy pixels before seam
            for x in 0..seam_x {
                result[dst_row_start + x] = image[src_row_start + x];
            }

            // Copy pixels after seam
            for x in seam_x + 1..width as usize {
                result[dst_row_start + x - 1] = image[src_row_start + x];
            }
        }

        Ok(result)
    }

    /// Remove a horizontal seam from a grayscale image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `seam` - Seam to remove
    ///
    /// # Returns
    ///
    /// Image with seam removed (height reduced by 1).
    pub fn remove_horizontal_seam(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        seam: &Seam,
    ) -> CvResult<Vec<u8>> {
        if seam.vertical {
            return Err(CvError::invalid_parameter(
                "seam",
                "expected horizontal seam",
            ));
        }

        if seam.path.len() != width as usize {
            return Err(CvError::invalid_parameter(
                "seam.path.len()",
                format!("expected {}, got {}", width, seam.path.len()),
            ));
        }

        let new_height = height - 1;
        let mut result = vec![0u8; width as usize * new_height as usize];

        for x in 0..width as usize {
            let seam_y = seam.path[x] as usize;
            let mut dst_y = 0;

            // Copy pixels before seam
            for y in 0..seam_y {
                result[dst_y * width as usize + x] = image[y * width as usize + x];
                dst_y += 1;
            }

            // Copy pixels after seam
            for y in seam_y + 1..height as usize {
                result[dst_y * width as usize + x] = image[y * width as usize + x];
                dst_y += 1;
            }
        }

        Ok(result)
    }

    /// Insert a vertical seam into a grayscale image.
    ///
    /// Duplicates pixels along the seam path to increase width.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `seam` - Seam to insert
    ///
    /// # Returns
    ///
    /// Image with seam inserted (width increased by 1).
    pub fn insert_vertical_seam(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        seam: &Seam,
    ) -> CvResult<Vec<u8>> {
        if !seam.vertical {
            return Err(CvError::invalid_parameter("seam", "expected vertical seam"));
        }

        if seam.path.len() != height as usize {
            return Err(CvError::invalid_parameter(
                "seam.path.len()",
                format!("expected {}, got {}", height, seam.path.len()),
            ));
        }

        let new_width = width + 1;
        let mut result = vec![0u8; new_width as usize * height as usize];

        for y in 0..height as usize {
            let seam_x = seam.path[y] as usize;
            let src_row_start = y * width as usize;
            let dst_row_start = y * new_width as usize;

            // Copy pixels before seam
            for x in 0..seam_x {
                result[dst_row_start + x] = image[src_row_start + x];
            }

            // Duplicate seam pixel
            result[dst_row_start + seam_x] = image[src_row_start + seam_x];

            // Average with right neighbor if available
            if seam_x < width as usize - 1 {
                let left = image[src_row_start + seam_x] as u16;
                let right = image[src_row_start + seam_x + 1] as u16;
                result[dst_row_start + seam_x + 1] = ((left + right) / 2) as u8;
            } else {
                result[dst_row_start + seam_x + 1] = image[src_row_start + seam_x];
            }

            // Copy remaining pixels
            for x in seam_x + 1..width as usize {
                result[dst_row_start + x + 1] = image[src_row_start + x];
            }
        }

        Ok(result)
    }

    /// Insert a horizontal seam into a grayscale image.
    ///
    /// Duplicates pixels along the seam path to increase height.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `seam` - Seam to insert
    ///
    /// # Returns
    ///
    /// Image with seam inserted (height increased by 1).
    pub fn insert_horizontal_seam(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        seam: &Seam,
    ) -> CvResult<Vec<u8>> {
        if seam.vertical {
            return Err(CvError::invalid_parameter(
                "seam",
                "expected horizontal seam",
            ));
        }

        if seam.path.len() != width as usize {
            return Err(CvError::invalid_parameter(
                "seam.path.len()",
                format!("expected {}, got {}", width, seam.path.len()),
            ));
        }

        let new_height = height + 1;
        let mut result = vec![0u8; width as usize * new_height as usize];

        for x in 0..width as usize {
            let seam_y = seam.path[x] as usize;
            let mut dst_y = 0;

            // Copy pixels before seam
            for y in 0..seam_y {
                result[dst_y * width as usize + x] = image[y * width as usize + x];
                dst_y += 1;
            }

            // Duplicate seam pixel
            result[dst_y * width as usize + x] = image[seam_y * width as usize + x];
            dst_y += 1;

            // Average with bottom neighbor if available
            if seam_y < height as usize - 1 {
                let top = image[seam_y * width as usize + x] as u16;
                let bottom = image[(seam_y + 1) * width as usize + x] as u16;
                result[dst_y * width as usize + x] = ((top + bottom) / 2) as u8;
            } else {
                result[dst_y * width as usize + x] = image[seam_y * width as usize + x];
            }
            dst_y += 1;

            // Copy remaining pixels
            for y in seam_y + 1..height as usize {
                result[dst_y * width as usize + x] = image[y * width as usize + x];
                dst_y += 1;
            }
        }

        Ok(result)
    }

    /// Resize image by removing vertical seams.
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image
    /// * `width` - Current width
    /// * `height` - Current height
    /// * `target_width` - Target width (must be less than current)
    ///
    /// # Returns
    ///
    /// Resized image.
    pub fn reduce_width(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        target_width: u32,
    ) -> CvResult<Vec<u8>> {
        if target_width >= width {
            return Err(CvError::invalid_parameter(
                "target_width",
                "must be less than current width",
            ));
        }

        let mut current_image = image.to_vec();
        let mut current_width = width;

        while current_width > target_width {
            let seam = self.find_vertical_seam(&current_image, current_width, height)?;
            current_image =
                self.remove_vertical_seam(&current_image, current_width, height, &seam)?;
            current_width -= 1;
        }

        Ok(current_image)
    }

    /// Resize image by removing horizontal seams.
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image
    /// * `width` - Current width
    /// * `height` - Current height
    /// * `target_height` - Target height (must be less than current)
    ///
    /// # Returns
    ///
    /// Resized image.
    pub fn reduce_height(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        target_height: u32,
    ) -> CvResult<Vec<u8>> {
        if target_height >= height {
            return Err(CvError::invalid_parameter(
                "target_height",
                "must be less than current height",
            ));
        }

        let mut current_image = image.to_vec();
        let mut current_height = height;

        while current_height > target_height {
            let seam = self.find_horizontal_seam(&current_image, width, current_height)?;
            current_image =
                self.remove_horizontal_seam(&current_image, width, current_height, &seam)?;
            current_height -= 1;
        }

        Ok(current_image)
    }

    /// Resize image by inserting vertical seams.
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image
    /// * `width` - Current width
    /// * `height` - Current height
    /// * `target_width` - Target width (must be greater than current)
    ///
    /// # Returns
    ///
    /// Resized image.
    pub fn enlarge_width(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        target_width: u32,
    ) -> CvResult<Vec<u8>> {
        if target_width <= width {
            return Err(CvError::invalid_parameter(
                "target_width",
                "must be greater than current width",
            ));
        }

        // Find all seams to insert first
        let num_seams = target_width - width;
        let mut seams = Vec::new();
        let mut temp_image = image.to_vec();
        let mut temp_width = width;

        for _ in 0..num_seams {
            let seam = self.find_vertical_seam(&temp_image, temp_width, height)?;
            seams.push(seam.clone());
            temp_image = self.remove_vertical_seam(&temp_image, temp_width, height, &seam)?;
            temp_width -= 1;
        }

        // Insert seams in order, adjusting positions
        let mut result = image.to_vec();
        let mut current_width = width;

        for (i, seam) in seams.iter().enumerate() {
            // Adjust seam positions based on previously inserted seams
            let mut adjusted_path = seam.path.clone();
            let path_len = adjusted_path.len();
            for idx in 0..path_len {
                let current_val = adjusted_path[idx];
                let mut offset = 0;
                for prev_seam in &seams[..i] {
                    let prev_path_len = prev_seam.path.len();
                    if prev_path_len > 0
                        && idx < prev_path_len
                        && current_val >= prev_seam.path[idx]
                    {
                        offset += 1;
                    }
                }
                adjusted_path[idx] += offset;
            }

            let adjusted_seam = Seam::new(adjusted_path, seam.energy, true);
            result = self.insert_vertical_seam(&result, current_width, height, &adjusted_seam)?;
            current_width += 1;
        }

        Ok(result)
    }

    /// Compute energy map for an image.
    fn compute_energy(&self, image: &[u8], width: u32, height: u32) -> CvResult<EnergyMap> {
        let energy_data = self.energy_function.compute(image, width, height)?;
        let mut energy_map = EnergyMap::from_data(energy_data, width, height)?;

        // Apply protection mask if set
        if let Some(ref mask) = self.protection_mask {
            energy_map.apply_protection_mask(mask, self.protection_scale);
        }

        Ok(energy_map)
    }
}

/// Find the minimum-energy vertical seam using dynamic programming.
fn find_min_vertical_seam(energy: &EnergyMap) -> Seam {
    let cumulative = compute_cumulative_energy(energy, true);
    let w = energy.width as usize;
    let h = energy.height as usize;

    // Find minimum in last row
    let last_row_start = (h - 1) * w;
    let (min_x, min_energy) = cumulative.data[last_row_start..last_row_start + w]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &e)| (i, e))
        // Safety: the slice has width `w` elements and `w` >= 1 (validated by EnergyMap construction)
        .expect("energy slice is non-empty (width >= 1 guaranteed by EnergyMap)");

    // Backtrack to find the seam path
    let mut path = vec![0u32; h];
    path[h - 1] = min_x as u32;

    for y in (0..h - 1).rev() {
        let x = path[y + 1] as usize;
        let mut min_prev_x = x;
        let mut min_prev_energy = cumulative.data[y * w + x];

        // Check left neighbor
        if x > 0 {
            let left_energy = cumulative.data[y * w + x - 1];
            if left_energy < min_prev_energy {
                min_prev_energy = left_energy;
                min_prev_x = x - 1;
            }
        }

        // Check right neighbor
        if x < w - 1 {
            let right_energy = cumulative.data[y * w + x + 1];
            if right_energy < min_prev_energy {
                min_prev_x = x + 1;
            }
        }

        path[y] = min_prev_x as u32;
    }

    Seam::new(path, min_energy, true)
}

/// Find the minimum-energy horizontal seam using dynamic programming.
fn find_min_horizontal_seam(energy: &EnergyMap) -> Seam {
    let cumulative = compute_cumulative_energy(energy, false);
    let w = energy.width as usize;
    let h = energy.height as usize;

    // Find minimum in last column
    let (min_y, min_energy) = (0..h)
        .map(|y| (y, cumulative.data[y * w + w - 1]))
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        // Safety: the iterator produces `h` elements and `h` >= 1 (validated by EnergyMap construction)
        .expect("energy column iterator is non-empty (height >= 1 guaranteed by EnergyMap)");

    // Backtrack to find the seam path
    let mut path = vec![0u32; w];
    path[w - 1] = min_y as u32;

    for x in (0..w - 1).rev() {
        let y = path[x + 1] as usize;
        let mut min_prev_y = y;
        let mut min_prev_energy = cumulative.data[y * w + x];

        // Check top neighbor
        if y > 0 {
            let top_energy = cumulative.data[(y - 1) * w + x];
            if top_energy < min_prev_energy {
                min_prev_energy = top_energy;
                min_prev_y = y - 1;
            }
        }

        // Check bottom neighbor
        if y < h - 1 {
            let bottom_energy = cumulative.data[(y + 1) * w + x];
            if bottom_energy < min_prev_energy {
                min_prev_y = y + 1;
            }
        }

        path[x] = min_prev_y as u32;
    }

    Seam::new(path, min_energy, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seam_new() {
        let seam = Seam::new(vec![0, 1, 2], 10.0, true);
        assert_eq!(seam.len(), 3);
        assert!(seam.vertical);
        assert_eq!(seam.energy, 10.0);
    }

    #[test]
    fn test_find_vertical_seam() {
        let image = vec![128u8; 100];
        let carver = SeamCarver::new(EnergyFunction::Gradient);
        let seam = carver
            .find_vertical_seam(&image, 10, 10)
            .expect("find_vertical_seam should succeed");
        assert_eq!(seam.len(), 10);
        assert!(seam.vertical);
    }

    #[test]
    fn test_remove_vertical_seam() {
        let image = vec![128u8; 100];
        let carver = SeamCarver::new(EnergyFunction::Gradient);
        let seam = carver
            .find_vertical_seam(&image, 10, 10)
            .expect("find_vertical_seam should succeed");
        let result = carver
            .remove_vertical_seam(&image, 10, 10, &seam)
            .expect("remove_vertical_seam should succeed");
        assert_eq!(result.len(), 90); // 9 x 10
    }

    #[test]
    fn test_insert_vertical_seam() {
        let image = vec![128u8; 100];
        let carver = SeamCarver::new(EnergyFunction::Gradient);
        let seam = carver
            .find_vertical_seam(&image, 10, 10)
            .expect("find_vertical_seam should succeed");
        let result = carver
            .insert_vertical_seam(&image, 10, 10, &seam)
            .expect("insert_vertical_seam should succeed");
        assert_eq!(result.len(), 110); // 11 x 10
    }

    #[test]
    fn test_reduce_width() {
        let image = vec![128u8; 100];
        let carver = SeamCarver::new(EnergyFunction::Gradient);
        let result = carver
            .reduce_width(&image, 10, 10, 8)
            .expect("reduce_width should succeed");
        assert_eq!(result.len(), 80); // 8 x 10
    }

    #[test]
    fn test_reduce_height() {
        let image = vec![128u8; 100];
        let carver = SeamCarver::new(EnergyFunction::Gradient);
        let result = carver
            .reduce_height(&image, 10, 10, 8)
            .expect("reduce_height should succeed");
        assert_eq!(result.len(), 80); // 10 x 8
    }
}
