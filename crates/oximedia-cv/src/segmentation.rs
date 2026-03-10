//! Image segmentation: connected components, region growing, and a
//! watershed-like approximation.
//!
//! All algorithms operate on flat, row-major grayscale `f32` or label `u32`
//! images.

/// A 2-D label map produced by segmentation algorithms.
#[derive(Debug, Clone)]
pub struct LabelMap {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Label for each pixel (0 = background / unlabelled).
    pub labels: Vec<u32>,
}

impl LabelMap {
    /// Create a new all-zero label map.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            labels: vec![0u32; width * height],
        }
    }

    /// Get the label at pixel `(x, y)`.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> u32 {
        self.labels[y * self.width + x]
    }

    /// Set the label at pixel `(x, y)`.
    pub fn set(&mut self, x: usize, y: usize, label: u32) {
        self.labels[y * self.width + x] = label;
    }

    /// Number of distinct non-zero labels.
    #[must_use]
    pub fn num_labels(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for &l in &self.labels {
            if l != 0 {
                seen.insert(l);
            }
        }
        seen.len()
    }

    /// Count pixels belonging to `label`.
    #[must_use]
    pub fn count_label(&self, label: u32) -> usize {
        self.labels.iter().filter(|&&l| l == label).count()
    }

    /// Bounding box of a label: `(x_min, y_min, x_max, y_max)`.
    #[must_use]
    pub fn bounding_box(&self, label: u32) -> Option<(usize, usize, usize, usize)> {
        let mut x_min = usize::MAX;
        let mut y_min = usize::MAX;
        let mut x_max = 0;
        let mut y_max = 0;
        let mut found = false;

        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) == label {
                    x_min = x_min.min(x);
                    y_min = y_min.min(y);
                    x_max = x_max.max(x);
                    y_max = y_max.max(y);
                    found = true;
                }
            }
        }

        found.then_some((x_min, y_min, x_max, y_max))
    }
}

/// Connected-component labeling using 4-connectivity (BFS / union-find hybrid).
///
/// Input: binary image where pixels > `threshold` are foreground.
#[must_use]
pub fn connected_components(
    image: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
) -> LabelMap {
    let mut map = LabelMap::new(width, height);
    if image.is_empty() || width == 0 || height == 0 {
        return map;
    }

    let mut current_label = 0u32;
    let mut stack = Vec::new();

    for sy in 0..height {
        for sx in 0..width {
            if image[sy * width + sx] <= threshold || map.get(sx, sy) != 0 {
                continue;
            }
            // New component
            current_label += 1;
            map.set(sx, sy, current_label);
            stack.push((sx, sy));

            while let Some((x, y)) = stack.pop() {
                // 4-connected neighbours
                let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dx, dy) in neighbours {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if image[ny * width + nx] > threshold && map.get(nx, ny) == 0 {
                        map.set(nx, ny, current_label);
                        stack.push((nx, ny));
                    }
                }
            }
        }
    }

    map
}

/// Region-growing segmentation starting from a seed pixel.
///
/// Grows while the absolute difference from the seed value is within
/// `tolerance`.
#[must_use]
pub fn region_growing(
    image: &[f32],
    width: usize,
    height: usize,
    seed_x: usize,
    seed_y: usize,
    tolerance: f32,
) -> LabelMap {
    let mut map = LabelMap::new(width, height);
    if image.is_empty() || seed_x >= width || seed_y >= height {
        return map;
    }

    let seed_val = image[seed_y * width + seed_x];
    let label = 1u32;
    let mut queue = std::collections::VecDeque::new();
    map.set(seed_x, seed_y, label);
    queue.push_back((seed_x, seed_y));

    while let Some((x, y)) = queue.pop_front() {
        let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dx, dy) in neighbours {
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                continue;
            }
            let nx = nx as usize;
            let ny = ny as usize;
            if map.get(nx, ny) == 0 {
                let val = image[ny * width + nx];
                if (val - seed_val).abs() <= tolerance {
                    map.set(nx, ny, label);
                    queue.push_back((nx, ny));
                }
            }
        }
    }

    map
}

/// Watershed-like segmentation approximation using distance-based flooding.
///
/// Seeds are provided as `(x, y)` pairs and each gets a unique label ≥ 1.
/// Pixels are flooded in order of increasing distance to any seed.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn watershed_approx(
    image: &[f32],
    width: usize,
    height: usize,
    seeds: &[(usize, usize)],
) -> LabelMap {
    let mut map = LabelMap::new(width, height);
    if image.is_empty() || seeds.is_empty() {
        return map;
    }

    // Priority queue: (distance * 1000 as u64, x, y, label)
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut heap: BinaryHeap<Reverse<(u64, usize, usize, u32)>> = BinaryHeap::new();

    for (i, &(sx, sy)) in seeds.iter().enumerate() {
        if sx < width && sy < height {
            let label = i as u32 + 1;
            map.set(sx, sy, label);
            heap.push(Reverse((0, sx, sy, label)));
        }
    }

    while let Some(Reverse((_, x, y, label))) = heap.pop() {
        let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dx, dy) in neighbours {
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                continue;
            }
            let nx = nx as usize;
            let ny = ny as usize;
            if map.get(nx, ny) == 0 {
                // Distance metric: image gradient magnitude (higher = harder to cross)
                let weight = (image[ny * width + nx] * 1000.0) as u64;
                map.set(nx, ny, label);
                heap.push(Reverse((weight, nx, ny, label)));
            }
        }
    }

    map
}

/// Simple mean-shift-inspired superpixel cluster (single pass, approximate).
///
/// Groups pixels whose value is within `bandwidth` of cluster centres.
/// Returns a label map (centre-based), capped at `max_clusters` clusters.
#[must_use]
pub fn mean_shift_simple(
    image: &[f32],
    width: usize,
    height: usize,
    bandwidth: f32,
    max_clusters: usize,
) -> LabelMap {
    let mut map = LabelMap::new(width, height);
    if image.is_empty() {
        return map;
    }

    let mut centres: Vec<f32> = Vec::new();
    let mut next_label = 1u32;

    for y in 0..height {
        for x in 0..width {
            let val = image[y * width + x];
            // Find nearest cluster centre
            let closest = centres
                .iter()
                .enumerate()
                .find(|(_, &c)| (c - val).abs() <= bandwidth);

            let label = if let Some((i, _)) = closest {
                i as u32 + 1
            } else if centres.len() < max_clusters {
                centres.push(val);
                let l = next_label;
                next_label += 1;
                l
            } else {
                // Assign to nearest centre regardless
                centres
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        ((*a - val).abs())
                            .partial_cmp(&((*b - val).abs()))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(1, |(i, _)| i as u32 + 1)
            };

            map.set(x, y, label);
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(val: f32, w: usize, h: usize) -> Vec<f32> {
        vec![val; w * h]
    }

    fn binary_image() -> (Vec<f32>, usize, usize) {
        // 5x5 image with two separate blobs
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            1.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        (data, 5, 5)
    }

    #[test]
    fn test_label_map_new() {
        let m = LabelMap::new(4, 4);
        assert_eq!(m.width, 4);
        assert_eq!(m.height, 4);
        assert!(m.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_label_map_set_get() {
        let mut m = LabelMap::new(5, 5);
        m.set(2, 3, 7);
        assert_eq!(m.get(2, 3), 7);
    }

    #[test]
    fn test_label_map_num_labels() {
        let mut m = LabelMap::new(3, 3);
        m.set(0, 0, 1);
        m.set(1, 1, 2);
        m.set(2, 2, 2);
        assert_eq!(m.num_labels(), 2);
    }

    #[test]
    fn test_label_map_count_label() {
        let mut m = LabelMap::new(3, 3);
        m.set(0, 0, 1);
        m.set(1, 1, 1);
        m.set(2, 2, 2);
        assert_eq!(m.count_label(1), 2);
        assert_eq!(m.count_label(2), 1);
        assert_eq!(m.count_label(3), 0);
    }

    #[test]
    fn test_label_map_bounding_box() {
        let mut m = LabelMap::new(10, 10);
        m.set(2, 3, 1);
        m.set(5, 7, 1);
        let bb = m.bounding_box(1).expect("bounding_box should succeed");
        assert_eq!(bb, (2, 3, 5, 7));
    }

    #[test]
    fn test_label_map_bounding_box_missing() {
        let m = LabelMap::new(5, 5);
        assert!(m.bounding_box(42).is_none());
    }

    #[test]
    fn test_connected_components_two_blobs() {
        let (img, w, h) = binary_image();
        let map = connected_components(&img, w, h, 0.5);
        assert_eq!(map.num_labels(), 2);
    }

    #[test]
    fn test_connected_components_blank_image() {
        let map = connected_components(&uniform(0.0, 5, 5), 5, 5, 0.5);
        assert_eq!(map.num_labels(), 0);
    }

    #[test]
    fn test_connected_components_full_image() {
        let map = connected_components(&uniform(1.0, 4, 4), 4, 4, 0.5);
        assert_eq!(map.num_labels(), 1);
    }

    #[test]
    fn test_region_growing_uniform() {
        let img = uniform(0.5, 6, 6);
        let map = region_growing(&img, 6, 6, 2, 2, 0.1);
        // All pixels should be in region 1
        assert_eq!(map.count_label(1), 36);
    }

    #[test]
    fn test_region_growing_limited() {
        let mut img = uniform(0.0, 6, 6);
        // Seed area with value 0.5, rest 0.0
        for x in 0..3 {
            img[0 * 6 + x] = 0.5;
            img[1 * 6 + x] = 0.5;
        }
        let map = region_growing(&img, 6, 6, 0, 0, 0.1);
        // Only the 0.5 patch should be included
        assert!(map.count_label(1) <= 6);
    }

    #[test]
    fn test_watershed_two_seeds() {
        let img = uniform(0.5, 10, 10);
        let seeds = vec![(1, 1), (8, 8)];
        let map = watershed_approx(&img, 10, 10, &seeds);
        assert_eq!(map.num_labels(), 2);
        // Every pixel should be labelled
        assert_eq!(map.labels.iter().filter(|&&l| l == 0).count(), 0);
    }

    #[test]
    fn test_watershed_no_seeds_returns_blank() {
        let img = uniform(0.5, 5, 5);
        let map = watershed_approx(&img, 5, 5, &[]);
        assert_eq!(map.num_labels(), 0);
    }

    #[test]
    fn test_mean_shift_uniform_image() {
        let img = uniform(0.5, 5, 5);
        let map = mean_shift_simple(&img, 5, 5, 0.1, 10);
        // All same value → single cluster
        assert_eq!(map.num_labels(), 1);
    }

    #[test]
    fn test_mean_shift_two_clusters() {
        let mut img = vec![0.0f32; 10];
        // First 5 pixels near 0.0, last 5 near 1.0
        for i in 5..10 {
            img[i] = 1.0;
        }
        let map = mean_shift_simple(&img, 10, 1, 0.1, 10);
        assert_eq!(map.num_labels(), 2);
    }
}
