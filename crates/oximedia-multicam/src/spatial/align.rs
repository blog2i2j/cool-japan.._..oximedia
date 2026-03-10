//! Spatial alignment for overlapping camera views.

use super::Transform2D;
use crate::{AngleId, Result};

/// Spatial aligner
#[derive(Debug)]
pub struct SpatialAligner {
    /// Transformations for each angle
    transforms: Vec<Transform2D>,
    /// Reference angle
    reference_angle: AngleId,
}

impl SpatialAligner {
    /// Create a new spatial aligner
    #[must_use]
    pub fn new(angle_count: usize, reference_angle: AngleId) -> Self {
        Self {
            transforms: vec![Transform2D::identity(); angle_count],
            reference_angle,
        }
    }

    /// Set reference angle
    pub fn set_reference_angle(&mut self, angle: AngleId) {
        self.reference_angle = angle;
    }

    /// Get reference angle
    #[must_use]
    pub fn reference_angle(&self) -> AngleId {
        self.reference_angle
    }

    /// Set transformation for angle
    pub fn set_transform(&mut self, angle: AngleId, transform: Transform2D) {
        if angle < self.transforms.len() {
            self.transforms[angle] = transform;
        }
    }

    /// Get transformation for angle
    #[must_use]
    pub fn get_transform(&self, angle: AngleId) -> Option<&Transform2D> {
        self.transforms.get(angle)
    }

    /// Apply alignment to point
    #[must_use]
    pub fn align_point(&self, angle: AngleId, x: f32, y: f32) -> (f32, f32) {
        if let Some(transform) = self.get_transform(angle) {
            transform.apply(x, y)
        } else {
            (x, y)
        }
    }

    /// Calculate alignment from feature points
    ///
    /// # Errors
    ///
    /// Returns an error if alignment fails
    pub fn calculate_alignment(
        &mut self,
        angle: AngleId,
        reference_points: &[(f32, f32)],
        target_points: &[(f32, f32)],
    ) -> Result<()> {
        if reference_points.len() != target_points.len() || reference_points.len() < 3 {
            return Err(crate::MultiCamError::SpatialAlignmentFailed(
                "Need at least 3 matching points".to_string(),
            ));
        }

        // Calculate centroid of both point sets
        let ref_centroid = Self::calculate_centroid(reference_points);
        let target_centroid = Self::calculate_centroid(target_points);

        // Calculate translation
        let tx = ref_centroid.0 - target_centroid.0;
        let ty = ref_centroid.1 - target_centroid.1;

        // Calculate scale (simplified - average distance from centroid)
        let ref_scale = Self::average_distance_from_centroid(reference_points, ref_centroid);
        let target_scale = Self::average_distance_from_centroid(target_points, target_centroid);

        let scale = if target_scale > 0.0 {
            ref_scale / target_scale
        } else {
            1.0
        };

        // Calculate rotation (simplified - using first point pair)
        let rotation = Self::calculate_rotation(
            reference_points[0],
            target_points[0],
            ref_centroid,
            target_centroid,
        );

        let transform = Transform2D {
            tx,
            ty,
            sx: scale,
            sy: scale,
            rotation,
        };

        self.set_transform(angle, transform);
        Ok(())
    }

    /// Calculate centroid of points
    fn calculate_centroid(points: &[(f32, f32)]) -> (f32, f32) {
        let sum = points
            .iter()
            .fold((0.0, 0.0), |acc, &p| (acc.0 + p.0, acc.1 + p.1));
        (sum.0 / points.len() as f32, sum.1 / points.len() as f32)
    }

    /// Calculate average distance from centroid
    fn average_distance_from_centroid(points: &[(f32, f32)], centroid: (f32, f32)) -> f32 {
        let sum: f32 = points
            .iter()
            .map(|&p| {
                let dx = p.0 - centroid.0;
                let dy = p.1 - centroid.1;
                (dx * dx + dy * dy).sqrt()
            })
            .sum();
        sum / points.len() as f32
    }

    /// Calculate rotation angle between two points relative to centroids
    fn calculate_rotation(
        ref_point: (f32, f32),
        target_point: (f32, f32),
        ref_centroid: (f32, f32),
        target_centroid: (f32, f32),
    ) -> f32 {
        let ref_angle = (ref_point.1 - ref_centroid.1).atan2(ref_point.0 - ref_centroid.0);
        let target_angle =
            (target_point.1 - target_centroid.1).atan2(target_point.0 - target_centroid.0);
        ref_angle - target_angle
    }

    /// Reset all transformations
    pub fn reset(&mut self) {
        for transform in &mut self.transforms {
            *transform = Transform2D::identity();
        }
    }

    /// Check if angle is aligned
    #[must_use]
    pub fn is_aligned(&self, angle: AngleId) -> bool {
        if let Some(transform) = self.get_transform(angle) {
            transform.tx != 0.0
                || transform.ty != 0.0
                || transform.rotation != 0.0
                || transform.sx != 1.0
                || transform.sy != 1.0
        } else {
            false
        }
    }
}

/// Alignment quality metrics
#[derive(Debug, Clone, Copy)]
pub struct AlignmentQuality {
    /// Alignment error (pixels)
    pub error: f32,
    /// Number of inliers
    pub inliers: usize,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
}

impl AlignmentQuality {
    /// Create new alignment quality metrics
    #[must_use]
    pub fn new(error: f32, inliers: usize, total_points: usize) -> Self {
        let confidence = if total_points > 0 {
            inliers as f32 / total_points as f32
        } else {
            0.0
        };

        Self {
            error,
            inliers,
            confidence,
        }
    }

    /// Check if alignment is good
    #[must_use]
    pub fn is_good(&self, max_error: f32, min_confidence: f32) -> bool {
        self.error < max_error && self.confidence >= min_confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligner_creation() {
        let aligner = SpatialAligner::new(3, 0);
        assert_eq!(aligner.reference_angle(), 0);
        assert_eq!(aligner.transforms.len(), 3);
    }

    #[test]
    fn test_set_transform() {
        let mut aligner = SpatialAligner::new(3, 0);
        let transform = Transform2D {
            tx: 10.0,
            ty: 20.0,
            ..Transform2D::identity()
        };

        aligner.set_transform(1, transform);
        let retrieved = aligner
            .get_transform(1)
            .expect("multicam test operation should succeed");
        assert_eq!(retrieved.tx, 10.0);
        assert_eq!(retrieved.ty, 20.0);
    }

    #[test]
    fn test_align_point() {
        let mut aligner = SpatialAligner::new(2, 0);
        let transform = Transform2D {
            tx: 5.0,
            ty: 10.0,
            ..Transform2D::identity()
        };
        aligner.set_transform(1, transform);

        let (x, y) = aligner.align_point(1, 10.0, 20.0);
        assert_eq!(x, 15.0);
        assert_eq!(y, 30.0);
    }

    #[test]
    fn test_calculate_centroid() {
        let points = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let centroid = SpatialAligner::calculate_centroid(&points);
        assert_eq!(centroid, (5.0, 5.0));
    }

    #[test]
    fn test_calculate_alignment() {
        let mut aligner = SpatialAligner::new(2, 0);

        let ref_points = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)];
        let target_points = vec![(5.0, 5.0), (15.0, 5.0), (15.0, 15.0)];

        let result = aligner.calculate_alignment(1, &ref_points, &target_points);
        assert!(result.is_ok());
        assert!(aligner.is_aligned(1));
    }

    #[test]
    fn test_alignment_quality() {
        let quality = AlignmentQuality::new(2.5, 45, 50);
        assert_eq!(quality.error, 2.5);
        assert_eq!(quality.inliers, 45);
        assert!((quality.confidence - 0.9).abs() < 0.01);
        assert!(quality.is_good(3.0, 0.8));
    }

    #[test]
    fn test_reset() {
        let mut aligner = SpatialAligner::new(2, 0);
        let transform = Transform2D {
            tx: 10.0,
            ty: 20.0,
            ..Transform2D::identity()
        };
        aligner.set_transform(1, transform);

        aligner.reset();
        let reset_transform = aligner
            .get_transform(1)
            .expect("multicam test operation should succeed");
        assert_eq!(reset_transform.tx, 0.0);
        assert_eq!(reset_transform.ty, 0.0);
    }
}
