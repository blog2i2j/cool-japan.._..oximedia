// Temporal coherence tracking for better temporal filtering.

/// Motion vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    /// Horizontal displacement.
    pub x: i32,
    /// Vertical displacement.
    pub y: i32,
    /// Confidence (0.0-1.0).
    pub confidence: f32,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub fn new(x: i32, y: i32, confidence: f32) -> Self {
        Self { x, y, confidence }
    }

    /// Get magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        ((self.x * self.x + self.y * self.y) as f32).sqrt()
    }

    /// Check if this is a zero vector.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.x == 0 && self.y == 0
    }
}

/// Motion field for a frame.
#[derive(Debug, Clone)]
pub struct MotionField {
    /// Width in blocks.
    pub width: u32,
    /// Height in blocks.
    pub height: u32,
    /// Block size.
    pub block_size: u32,
    /// Motion vectors.
    pub vectors: Vec<MotionVector>,
}

impl MotionField {
    /// Create a new motion field.
    #[must_use]
    pub fn new(frame_width: u32, frame_height: u32, block_size: u32) -> Self {
        let width = frame_width.div_ceil(block_size);
        let height = frame_height.div_ceil(block_size);
        let vectors = vec![MotionVector::new(0, 0, 1.0); (width * height) as usize];

        Self {
            width,
            height,
            block_size,
            vectors,
        }
    }

    /// Get motion vector for a block.
    #[must_use]
    pub fn get(&self, bx: u32, by: u32) -> MotionVector {
        if bx >= self.width || by >= self.height {
            return MotionVector::new(0, 0, 0.0);
        }
        self.vectors[(by * self.width + bx) as usize]
    }

    /// Set motion vector for a block.
    pub fn set(&mut self, bx: u32, by: u32, mv: MotionVector) {
        if bx < self.width && by < self.height {
            self.vectors[(by * self.width + bx) as usize] = mv;
        }
    }

    /// Get average motion magnitude.
    #[must_use]
    pub fn average_magnitude(&self) -> f32 {
        let sum: f32 = self.vectors.iter().map(|mv| mv.magnitude()).sum();
        sum / self.vectors.len() as f32
    }

    /// Smooth motion field.
    pub fn smooth(&mut self, iterations: usize) {
        for _ in 0..iterations {
            let original = self.vectors.clone();

            for by in 0..self.height {
                for bx in 0..self.width {
                    let mut sum_x = 0.0f32;
                    let mut sum_y = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = bx as i32 + dx;
                            let ny = by as i32 + dy;

                            if nx >= 0
                                && nx < self.width as i32
                                && ny >= 0
                                && ny < self.height as i32
                            {
                                let idx = (ny as u32 * self.width + nx as u32) as usize;
                                if let Some(mv) = original.get(idx) {
                                    let weight = mv.confidence;
                                    sum_x += mv.x as f32 * weight;
                                    sum_y += mv.y as f32 * weight;
                                    weight_sum += weight;
                                }
                            }
                        }
                    }

                    if weight_sum > 0.0 {
                        let idx = (by * self.width + bx) as usize;
                        self.vectors[idx].x = (sum_x / weight_sum).round() as i32;
                        self.vectors[idx].y = (sum_y / weight_sum).round() as i32;
                    }
                }
            }
        }
    }
}

/// Temporal coherence tracker.
#[derive(Debug)]
pub struct CoherenceTracker {
    /// Motion fields for recent frames.
    motion_history: Vec<MotionField>,
    /// Maximum history length.
    max_history: usize,
}

impl CoherenceTracker {
    /// Create a new coherence tracker.
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            motion_history: Vec::new(),
            max_history,
        }
    }

    /// Add a motion field.
    pub fn add_motion_field(&mut self, field: MotionField) {
        self.motion_history.push(field);
        if self.motion_history.len() > self.max_history {
            self.motion_history.remove(0);
        }
    }

    /// Get motion consistency at a location.
    #[must_use]
    pub fn get_consistency(&self, bx: u32, by: u32) -> f32 {
        if self.motion_history.len() < 2 {
            return 1.0;
        }

        let mut total_diff = 0.0f32;
        let mut count = 0;

        for i in 1..self.motion_history.len() {
            let prev_mv = self.motion_history[i - 1].get(bx, by);
            let curr_mv = self.motion_history[i].get(bx, by);

            let dx = (curr_mv.x - prev_mv.x) as f32;
            let dy = (curr_mv.y - prev_mv.y) as f32;
            let diff = (dx * dx + dy * dy).sqrt();

            total_diff += diff;
            count += 1;
        }

        if count > 0 {
            let avg_diff = total_diff / count as f32;
            (1.0 / (1.0 + avg_diff * 0.1)).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Clear history.
    pub fn clear(&mut self) {
        self.motion_history.clear();
    }
}
