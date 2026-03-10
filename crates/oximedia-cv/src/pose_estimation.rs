//! Human pose estimation module.
//!
//! Provides joint keypoint detection and skeleton construction from heatmaps.

/// Represents a body joint type in the COCO-style pose skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum JointType {
    /// Nose keypoint
    Nose,
    /// Left eye keypoint
    LeftEye,
    /// Right eye keypoint
    RightEye,
    /// Left ear keypoint
    LeftEar,
    /// Right ear keypoint
    RightEar,
    /// Left shoulder keypoint
    LeftShoulder,
    /// Right shoulder keypoint
    RightShoulder,
    /// Left elbow keypoint
    LeftElbow,
    /// Right elbow keypoint
    RightElbow,
    /// Left wrist keypoint
    LeftWrist,
    /// Right wrist keypoint
    RightWrist,
    /// Left hip keypoint
    LeftHip,
    /// Right hip keypoint
    RightHip,
    /// Left knee keypoint
    LeftKnee,
    /// Right knee keypoint
    RightKnee,
    /// Left ankle keypoint
    LeftAnkle,
    /// Right ankle keypoint
    RightAnkle,
}

impl JointType {
    /// Returns the numeric index for this joint type (COCO ordering).
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Nose => 0,
            Self::LeftEye => 1,
            Self::RightEye => 2,
            Self::LeftEar => 3,
            Self::RightEar => 4,
            Self::LeftShoulder => 5,
            Self::RightShoulder => 6,
            Self::LeftElbow => 7,
            Self::RightElbow => 8,
            Self::LeftWrist => 9,
            Self::RightWrist => 10,
            Self::LeftHip => 11,
            Self::RightHip => 12,
            Self::LeftKnee => 13,
            Self::RightKnee => 14,
            Self::LeftAnkle => 15,
            Self::RightAnkle => 16,
        }
    }

    /// Returns `true` if this joint is on the face.
    #[must_use]
    pub fn is_face(self) -> bool {
        matches!(
            self,
            Self::Nose | Self::LeftEye | Self::RightEye | Self::LeftEar | Self::RightEar
        )
    }

    /// Returns `true` if this joint belongs to the upper body (shoulders, elbows, wrists).
    #[must_use]
    pub fn is_upper_body(self) -> bool {
        matches!(
            self,
            Self::LeftShoulder
                | Self::RightShoulder
                | Self::LeftElbow
                | Self::RightElbow
                | Self::LeftWrist
                | Self::RightWrist
        )
    }
}

/// A detected keypoint for a single joint.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct KeyPoint {
    /// The joint type this keypoint represents.
    pub joint: JointType,
    /// Horizontal coordinate in image pixels.
    pub x: f32,
    /// Vertical coordinate in image pixels.
    pub y: f32,
    /// Detection confidence score in `[0, 1]`.
    pub confidence: f32,
}

impl KeyPoint {
    /// Returns `true` if the keypoint confidence exceeds `threshold`.
    #[must_use]
    pub fn is_visible(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// A full-body pose skeleton composed of keypoints.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PoseSkeleton {
    /// All keypoints belonging to this skeleton.
    pub keypoints: Vec<KeyPoint>,
}

impl PoseSkeleton {
    /// Computes the approximate head center as the average position of face keypoints
    /// that are visible above a fixed threshold of 0.1.
    #[must_use]
    pub fn head_center(&self) -> Option<(f32, f32)> {
        let face_pts: Vec<&KeyPoint> = self
            .keypoints
            .iter()
            .filter(|kp| kp.joint.is_face() && kp.is_visible(0.1))
            .collect();
        if face_pts.is_empty() {
            return None;
        }
        let n = face_pts.len() as f32;
        let x = face_pts.iter().map(|kp| kp.x).sum::<f32>() / n;
        let y = face_pts.iter().map(|kp| kp.y).sum::<f32>() / n;
        Some((x, y))
    }

    /// Computes the hip center as the midpoint between left and right hips when both
    /// are visible above a fixed threshold of 0.1.
    #[must_use]
    pub fn hip_center(&self) -> Option<(f32, f32)> {
        let find = |joint: JointType| {
            self.keypoints
                .iter()
                .find(|kp| kp.joint == joint && kp.is_visible(0.1))
        };
        let lh = find(JointType::LeftHip)?;
        let rh = find(JointType::RightHip)?;
        Some(((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0))
    }

    /// Returns references to all keypoints whose confidence is at least `threshold`.
    #[must_use]
    pub fn visible_keypoints(&self, threshold: f32) -> Vec<&KeyPoint> {
        self.keypoints
            .iter()
            .filter(|kp| kp.is_visible(threshold))
            .collect()
    }

    /// Returns `true` when every joint in the skeleton is visible above `threshold`.
    #[must_use]
    pub fn is_complete(&self, threshold: f32) -> bool {
        self.keypoints.len() == 17 && self.keypoints.iter().all(|kp| kp.is_visible(threshold))
    }

    /// Returns the axis-aligned bounding box `(x, y, width, height)` enclosing all
    /// keypoints with confidence above 0.1.
    #[must_use]
    pub fn bounding_box(&self) -> (f32, f32, f32, f32) {
        let visible: Vec<&KeyPoint> = self
            .keypoints
            .iter()
            .filter(|kp| kp.is_visible(0.1))
            .collect();
        if visible.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let min_x = visible.iter().map(|kp| kp.x).fold(f32::MAX, f32::min);
        let min_y = visible.iter().map(|kp| kp.y).fold(f32::MAX, f32::min);
        let max_x = visible.iter().map(|kp| kp.x).fold(f32::MIN, f32::max);
        let max_y = visible.iter().map(|kp| kp.y).fold(f32::MIN, f32::max);
        (min_x, min_y, max_x - min_x, max_y - min_y)
    }
}

/// Performs peak-finding pose estimation on a set of heatmaps.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct PoseEstimator;

impl PoseEstimator {
    /// Detects poses from a slice of heatmaps (one per joint channel).
    ///
    /// Each heatmap is a flattened `width * height` grid of confidence scores.
    /// Returns a single `PoseSkeleton` per detected person (simplified: one person).
    #[must_use]
    pub fn detect(heatmaps: &[Vec<f32>], width: usize, height: usize) -> Vec<PoseSkeleton> {
        if heatmaps.is_empty() || width == 0 || height == 0 {
            return Vec::new();
        }

        let joints = [
            JointType::Nose,
            JointType::LeftEye,
            JointType::RightEye,
            JointType::LeftEar,
            JointType::RightEar,
            JointType::LeftShoulder,
            JointType::RightShoulder,
            JointType::LeftElbow,
            JointType::RightElbow,
            JointType::LeftWrist,
            JointType::RightWrist,
            JointType::LeftHip,
            JointType::RightHip,
            JointType::LeftKnee,
            JointType::RightKnee,
            JointType::LeftAnkle,
            JointType::RightAnkle,
        ];

        let mut keypoints = Vec::new();

        for (channel_idx, joint) in joints.iter().enumerate() {
            let Some(heatmap) = heatmaps.get(channel_idx) else {
                keypoints.push(KeyPoint {
                    joint: *joint,
                    x: 0.0,
                    y: 0.0,
                    confidence: 0.0,
                });
                continue;
            };

            let (peak_idx, peak_val) = find_peak(heatmap);
            let px = (peak_idx % width) as f32;
            let py = (peak_idx / width) as f32;

            keypoints.push(KeyPoint {
                joint: *joint,
                x: px,
                y: py,
                confidence: peak_val,
            });
        }

        // Only return a skeleton if at least one keypoint is above threshold
        let any_confident = keypoints.iter().any(|kp| kp.confidence > 0.1);
        if any_confident {
            vec![PoseSkeleton { keypoints }]
        } else {
            Vec::new()
        }
    }
}

/// Finds the index and value of the maximum element in a slice.
fn find_peak(heatmap: &[f32]) -> (usize, f32) {
    heatmap
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(best_idx, best_val), (i, &v)| {
            if v > best_val {
                (i, v)
            } else {
                (best_idx, best_val)
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_joint(joint: JointType, x: f32, y: f32, confidence: f32) -> KeyPoint {
        KeyPoint {
            joint,
            x,
            y,
            confidence,
        }
    }

    // ---- JointType tests ----

    #[test]
    fn test_joint_index_nose() {
        assert_eq!(JointType::Nose.index(), 0);
    }

    #[test]
    fn test_joint_index_right_ankle() {
        assert_eq!(JointType::RightAnkle.index(), 16);
    }

    #[test]
    fn test_joint_is_face_true() {
        assert!(JointType::LeftEye.is_face());
        assert!(JointType::RightEar.is_face());
        assert!(JointType::Nose.is_face());
    }

    #[test]
    fn test_joint_is_face_false() {
        assert!(!JointType::LeftShoulder.is_face());
        assert!(!JointType::LeftAnkle.is_face());
    }

    #[test]
    fn test_joint_is_upper_body() {
        assert!(JointType::LeftShoulder.is_upper_body());
        assert!(JointType::RightWrist.is_upper_body());
        assert!(!JointType::LeftHip.is_upper_body());
        assert!(!JointType::Nose.is_upper_body());
    }

    // ---- KeyPoint tests ----

    #[test]
    fn test_keypoint_is_visible_above_threshold() {
        let kp = make_joint(JointType::Nose, 10.0, 20.0, 0.8);
        assert!(kp.is_visible(0.5));
        assert!(!kp.is_visible(0.9));
    }

    #[test]
    fn test_keypoint_is_visible_at_threshold_boundary() {
        let kp = make_joint(JointType::Nose, 0.0, 0.0, 0.5);
        assert!(kp.is_visible(0.5));
        assert!(!kp.is_visible(0.500_001));
    }

    // ---- PoseSkeleton tests ----

    fn make_full_skeleton(confidence: f32) -> PoseSkeleton {
        let joints = [
            JointType::Nose,
            JointType::LeftEye,
            JointType::RightEye,
            JointType::LeftEar,
            JointType::RightEar,
            JointType::LeftShoulder,
            JointType::RightShoulder,
            JointType::LeftElbow,
            JointType::RightElbow,
            JointType::LeftWrist,
            JointType::RightWrist,
            JointType::LeftHip,
            JointType::RightHip,
            JointType::LeftKnee,
            JointType::RightKnee,
            JointType::LeftAnkle,
            JointType::RightAnkle,
        ];
        let keypoints = joints
            .iter()
            .enumerate()
            .map(|(i, &joint)| KeyPoint {
                joint,
                x: i as f32 * 10.0,
                y: i as f32 * 5.0,
                confidence,
            })
            .collect();
        PoseSkeleton { keypoints }
    }

    #[test]
    fn test_skeleton_is_complete_all_confident() {
        let sk = make_full_skeleton(0.9);
        assert!(sk.is_complete(0.5));
    }

    #[test]
    fn test_skeleton_is_complete_low_confidence() {
        let sk = make_full_skeleton(0.3);
        assert!(!sk.is_complete(0.5));
    }

    #[test]
    fn test_skeleton_visible_keypoints_count() {
        let sk = make_full_skeleton(0.8);
        assert_eq!(sk.visible_keypoints(0.5).len(), 17);
        assert_eq!(sk.visible_keypoints(0.9).len(), 0);
    }

    #[test]
    fn test_skeleton_head_center_all_visible() {
        let sk = make_full_skeleton(0.9);
        let center = sk.head_center();
        assert!(center.is_some());
        let (cx, cy) = center.expect("operation should succeed");
        // Nose(0,0), LeftEye(10,5), RightEye(20,10), LeftEar(30,15), RightEar(40,20)
        assert!((cx - 20.0).abs() < 1e-4);
        assert!((cy - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_skeleton_head_center_none_when_invisible() {
        let sk = make_full_skeleton(0.0);
        assert!(sk.head_center().is_none());
    }

    #[test]
    fn test_skeleton_hip_center() {
        let mut sk = make_full_skeleton(0.9);
        // LeftHip index 11 -> x=110, y=55; RightHip index 12 -> x=120, y=60
        let center = sk.hip_center();
        assert!(center.is_some());
        let (cx, cy) = center.expect("operation should succeed");
        assert!((cx - 115.0).abs() < 1e-4);
        assert!((cy - 57.5).abs() < 1e-4);

        // Make hips invisible
        for kp in &mut sk.keypoints {
            if kp.joint == JointType::LeftHip || kp.joint == JointType::RightHip {
                kp.confidence = 0.0;
            }
        }
        assert!(sk.hip_center().is_none());
    }

    #[test]
    fn test_skeleton_bounding_box_empty() {
        let sk = PoseSkeleton { keypoints: vec![] };
        assert_eq!(sk.bounding_box(), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_skeleton_bounding_box_single_point() {
        let sk = PoseSkeleton {
            keypoints: vec![make_joint(JointType::Nose, 5.0, 7.0, 0.9)],
        };
        let (x, y, w, h) = sk.bounding_box();
        assert!((x - 5.0).abs() < 1e-4);
        assert!((y - 7.0).abs() < 1e-4);
        assert!((w).abs() < 1e-4);
        assert!((h).abs() < 1e-4);
    }

    // ---- PoseEstimator tests ----

    #[test]
    fn test_detect_empty_heatmaps() {
        let result = PoseEstimator::detect(&[], 10, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_zero_dimensions() {
        let heatmap = vec![vec![1.0_f32; 100]; 17];
        let result = PoseEstimator::detect(&heatmap, 0, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_finds_peak() {
        // 17 channels, each 4x4 = 16 values, peak at position 5 (x=1,y=1)
        let mut heatmaps: Vec<Vec<f32>> = vec![vec![0.0; 16]; 17];
        for ch in &mut heatmaps {
            ch[5] = 0.9;
        }
        let result = PoseEstimator::detect(&heatmaps, 4, 4);
        assert_eq!(result.len(), 1);
        let nose = &result[0].keypoints[0];
        assert_eq!(nose.joint, JointType::Nose);
        assert!((nose.x - 1.0).abs() < 1e-4); // 5 % 4 = 1
        assert!((nose.y - 1.0).abs() < 1e-4); // 5 / 4 = 1
        assert!((nose.confidence - 0.9).abs() < 1e-4);
    }

    #[test]
    fn test_detect_all_zero_returns_empty() {
        let heatmaps: Vec<Vec<f32>> = vec![vec![0.0; 16]; 17];
        let result = PoseEstimator::detect(&heatmaps, 4, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_fewer_channels_than_joints() {
        // Only 5 heatmap channels; remaining joints get confidence 0
        let heatmaps: Vec<Vec<f32>> = vec![vec![0.9_f32; 16]; 5];
        let result = PoseEstimator::detect(&heatmaps, 4, 4);
        assert_eq!(result.len(), 1);
        // Joints 0-4 should have confidence 0.9, 5-16 should have confidence 0.0
        assert!((result[0].keypoints[0].confidence - 0.9).abs() < 1e-4);
        assert!((result[0].keypoints[5].confidence).abs() < 1e-4);
    }
}
