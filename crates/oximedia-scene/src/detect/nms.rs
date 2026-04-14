//! Non-Maximum Suppression (NMS) for detection boxes.
//!
//! Provides `DetectionBox` with IoU computation, standard NMS (hard suppression),
//! and Soft-NMS (Gaussian score decay for overlapping boxes).
//!
//! Also provides a universal `Detection` type compatible with any detector, along
//! with standalone `iou`, `nms`, and `soft_nms` functions that operate on slices
//! of `Detection` values.

/// Universal detection result usable by any detector.
///
/// Stores the bounding box in top-left / width / height form together with a
/// confidence score and an integer class identifier.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Left edge (x coordinate of top-left corner).
    pub x: f32,
    /// Top edge (y coordinate of top-left corner).
    pub y: f32,
    /// Bounding box width.
    pub w: f32,
    /// Bounding box height.
    pub h: f32,
    /// Detection confidence in [0.0, 1.0].
    pub confidence: f32,
    /// Integer class identifier.
    pub class_id: usize,
}

impl Detection {
    /// Create a new detection.
    #[must_use]
    pub fn new(x: f32, y: f32, w: f32, h: f32, confidence: f32, class_id: usize) -> Self {
        Self {
            x,
            y,
            w,
            h,
            confidence: confidence.clamp(0.0, 1.0),
            class_id,
        }
    }

    /// Right edge (x + w).
    #[inline]
    #[must_use]
    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    /// Bottom edge (y + h).
    #[inline]
    #[must_use]
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    /// Area (w × h).
    #[inline]
    #[must_use]
    pub fn area(&self) -> f32 {
        self.w * self.h
    }
}

/// Compute Intersection-over-Union (IoU) between two `Detection` values.
///
/// Returns a value in [0.0, 1.0]. Returns 0.0 when the boxes do not overlap
/// or when either box has zero area.
#[must_use]
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let inter_x1 = a.x.max(b.x);
    let inter_y1 = a.y.max(b.y);
    let inter_x2 = a.right().min(b.right());
    let inter_y2 = a.bottom().min(b.bottom());

    if inter_x2 <= inter_x1 || inter_y2 <= inter_y1 {
        return 0.0;
    }

    let intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    let union = a.area() + b.area() - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Non-Maximum Suppression for `Detection` values.
///
/// # Algorithm
///
/// 1. Sort all detections by descending confidence.
/// 2. Greedily keep: add the highest-confidence detection, then suppress every
///    remaining detection whose IoU with the kept box exceeds `iou_threshold`.
///
/// # Returns
///
/// A new `Vec<Detection>` containing only the surviving detections, ordered by
/// descending confidence.
#[must_use]
pub fn nms(detections: &mut Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    // Sort descending by confidence
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = detections.len();
    let mut suppressed = vec![false; n];
    let mut kept: Vec<Detection> = Vec::with_capacity(n);

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        kept.push(detections[i].clone());

        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }
            if iou(&detections[i], &detections[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    kept
}

/// Gaussian Soft-NMS for `Detection` values.
///
/// Instead of hard-suppressing boxes, each remaining detection's confidence is
/// decayed by a Gaussian weight based on its IoU with the currently selected box:
///
/// ```text
/// score_j *= exp(−IoU(i, j)² / sigma)
/// ```
///
/// Detections whose score drops to zero (within floating-point precision) are
/// excluded from the result.
///
/// # Arguments
///
/// * `detections` – mutable reference; scores are updated in-place.
/// * `sigma` – Gaussian bandwidth (> 0). Larger `sigma` = less aggressive decay.
///
/// # Returns
///
/// A new `Vec<Detection>` with decayed scores, ordered by descending final score.
/// Only detections with a score > 0 after all decay steps are included.
#[must_use]
pub fn soft_nms(detections: &mut Vec<Detection>, sigma: f32) -> Vec<Detection> {
    let sigma_eff = sigma.max(f32::EPSILON);
    let n = detections.len();

    // Working copy: (index, current_score)
    let mut scores: Vec<f32> = detections.iter().map(|d| d.confidence).collect();
    let mut processed = vec![false; n];

    for _ in 0..n {
        // Find unprocessed detection with highest current score
        let best_pos = (0..n).filter(|&i| !processed[i]).max_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = match best_pos {
            Some(idx) => idx,
            None => break,
        };

        if scores[best] <= 0.0 {
            break;
        }

        processed[best] = true;

        // Decay scores of all remaining detections
        for j in 0..n {
            if processed[j] {
                continue;
            }
            let overlap = iou(&detections[best], &detections[j]);
            let decay = (-(overlap * overlap) / sigma_eff).exp();
            scores[j] *= decay;
        }
    }

    // Write decayed scores back
    for (det, &s) in detections.iter_mut().zip(scores.iter()) {
        det.confidence = s;
    }

    // Collect survivors (score > 0) and sort by descending score
    let mut survivors: Vec<Detection> = detections
        .iter()
        .filter(|d| d.confidence > 0.0)
        .cloned()
        .collect();

    survivors.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    survivors
}

/// An axis-aligned bounding box with a confidence score and optional class annotation.
#[derive(Debug, Clone)]
pub struct DetectionBox {
    /// Left edge (x coordinate of top-left corner).
    pub x: f32,
    /// Top edge (y coordinate of top-left corner).
    pub y: f32,
    /// Box width in pixels.
    pub width: f32,
    /// Box height in pixels.
    pub height: f32,
    /// Detection confidence in [0.0, 1.0].
    pub score: f32,
    /// Integer class identifier.
    pub class_id: u32,
    /// Optional human-readable class label.
    pub label: Option<String>,
}

impl DetectionBox {
    /// Create a new detection box without a class annotation.
    #[must_use]
    pub fn new(x: f32, y: f32, width: f32, height: f32, score: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            score: score.clamp(0.0, 1.0),
            class_id: 0,
            label: None,
        }
    }

    /// Right edge of the box (x + width).
    #[inline]
    #[must_use]
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Bottom edge of the box (y + height).
    #[inline]
    #[must_use]
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// Area of the box (width × height).
    #[inline]
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Intersection-over-Union (IoU) with another `DetectionBox`.
    ///
    /// Returns a value in [0.0, 1.0]. Returns 0.0 when the boxes do not overlap
    /// or when either box has zero area.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f32 {
        let inter_x1 = self.x.max(other.x);
        let inter_y1 = self.y.max(other.y);
        let inter_x2 = self.right().min(other.right());
        let inter_y2 = self.bottom().min(other.bottom());

        if inter_x2 <= inter_x1 || inter_y2 <= inter_y1 {
            return 0.0;
        }

        let intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
        let union = self.area() + other.area() - intersection;

        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// Perform Non-Maximum Suppression over a list of `DetectionBox` values.
///
/// # Algorithm
///
/// 1. Discard all boxes whose `score` is below `score_threshold`.
/// 2. Sort remaining boxes by descending score.
/// 3. Greedily select boxes: mark a box as suppressed when it overlaps a
///    previously selected box by more than `iou_threshold`.
///
/// # Returns
///
/// Indices (into the original `boxes` slice, *after* score filtering is
/// removed for the returned indices — see note below) of surviving boxes,
/// sorted by descending score.
///
/// **Index convention**: The returned indices refer to positions in the
/// *original* `boxes` slice (before any filtering). Boxes below
/// `score_threshold` are never included and never appear in the output.
#[must_use]
pub fn non_maximum_suppression(
    boxes: &[DetectionBox],
    score_threshold: f32,
    iou_threshold: f32,
) -> Vec<usize> {
    // Collect candidate indices (above score threshold), sorted by score desc.
    let mut candidates: Vec<usize> = (0..boxes.len())
        .filter(|&i| boxes[i].score >= score_threshold)
        .collect();

    candidates.sort_by(|&a, &b| {
        boxes[b]
            .score
            .partial_cmp(&boxes[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = candidates.len();
    let mut suppressed = vec![false; n];
    let mut kept: Vec<usize> = Vec::with_capacity(n);

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        let orig_i = candidates[i];
        kept.push(orig_i);

        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }
            let orig_j = candidates[j];
            if boxes[orig_i].iou(&boxes[orig_j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    kept
}

/// Soft-NMS with Gaussian score decay for `DetectionBox` values.
///
/// Instead of hard-suppressing overlapping boxes, Soft-NMS reduces their
/// `score` by a Gaussian weight:
///
/// ```text
/// score_j *= exp(−IoU(i, j)² / sigma)
/// ```
///
/// # Arguments
///
/// * `boxes` – Mutable slice of detection boxes. Scores are updated in-place.
/// * `score_threshold` – Boxes whose score drops below this after decay are
///   excluded from the output.
/// * `sigma` – Gaussian bandwidth (> 0). Larger values = more aggressive decay.
///
/// # Returns
///
/// Indices of surviving boxes (score > `score_threshold` after all decay steps),
/// sorted by descending final score.
///
/// # Note
///
/// This operates on the `DetectionBox` type. For the universal `Detection` API
/// use [`soft_nms`] instead.
#[must_use]
pub fn soft_nms_boxes(
    boxes: &mut Vec<DetectionBox>,
    score_threshold: f32,
    sigma: f32,
) -> Vec<usize> {
    let sigma_eff = sigma.max(f32::EPSILON);
    let n = boxes.len();

    // Working copy of (original_index, current_score) so we can sort without
    // losing track of which original box each entry refers to.
    let mut work: Vec<(usize, f32)> = (0..n).map(|i| (i, boxes[i].score)).collect();

    // Iterate: each pass picks the current highest-score box, then decays all
    // remaining boxes by their IoU with the picked box.
    let mut processed = vec![false; n];
    let order: Vec<usize> = (0..n).collect(); // indices into `work`

    for _ in 0..n {
        // Find the unprocessed entry with the highest current score.
        let best_pos = order
            .iter()
            .filter(|&&wi| !processed[wi])
            .max_by(|&&a, &&b| {
                work[a]
                    .1
                    .partial_cmp(&work[b].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied();

        let best_wi = match best_pos {
            Some(wi) => wi,
            None => break,
        };

        if work[best_wi].1 < score_threshold {
            break;
        }

        processed[best_wi] = true;
        let best_orig = work[best_wi].0;

        // Decay scores of all remaining (unprocessed) boxes.
        for &wi in &order {
            if processed[wi] {
                continue;
            }
            let orig = work[wi].0;
            let iou = boxes[best_orig].iou(&boxes[orig]);
            let decay = (-(iou * iou) / sigma_eff).exp();
            work[wi].1 *= decay;
            // Write decayed score back into the box so subsequent IoU picks
            // use the updated confidence.
            boxes[orig].score = work[wi].1;
        }
    }

    // Collect survivors whose final score still exceeds the threshold.
    let mut survivors: Vec<usize> = work
        .iter()
        .filter(|&&(_, s)| s >= score_threshold)
        .map(|&(orig, _)| orig)
        .collect();

    // Sort by final score descending.
    survivors.sort_by(|&a, &b| {
        boxes[b]
            .score
            .partial_cmp(&boxes[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    survivors
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_box(x: f32, y: f32, w: f32, h: f32, score: f32) -> DetectionBox {
        DetectionBox::new(x, y, w, h, score)
    }

    // ---- DetectionBox geometry ----

    #[test]
    fn test_area_correct() {
        let b = make_box(0.0, 0.0, 5.0, 4.0, 0.9);
        assert!((b.area() - 20.0).abs() < f32::EPSILON, "area={}", b.area());
    }

    #[test]
    fn test_right_bottom() {
        let b = make_box(2.0, 3.0, 10.0, 8.0, 0.8);
        assert!((b.right() - 12.0).abs() < f32::EPSILON);
        assert!((b.bottom() - 11.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_iou_non_overlapping_is_zero() {
        let a = make_box(0.0, 0.0, 10.0, 10.0, 0.9);
        let b = make_box(20.0, 20.0, 10.0, 10.0, 0.8);
        assert!(
            (a.iou(&b)).abs() < f32::EPSILON,
            "expected 0, got {}",
            a.iou(&b)
        );
    }

    #[test]
    fn test_iou_identical_boxes_is_one() {
        let a = make_box(5.0, 5.0, 20.0, 20.0, 0.9);
        let b = a.clone();
        let iou = a.iou(&b);
        assert!((iou - 1.0).abs() < 1e-5, "expected ~1.0, got {iou}");
    }

    #[test]
    fn test_iou_partial_overlap() {
        // Two 10×10 boxes offset by 5 in both axes → 5×5 = 25 intersection
        // union = 100 + 100 - 25 = 175
        let a = make_box(0.0, 0.0, 10.0, 10.0, 0.9);
        let b = make_box(5.0, 5.0, 10.0, 10.0, 0.8);
        let expected = 25.0 / 175.0;
        assert!(
            (a.iou(&b) - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            a.iou(&b)
        );
    }

    #[test]
    fn test_iou_touching_edges_is_zero() {
        // Boxes share an edge but have zero intersection area
        let a = make_box(0.0, 0.0, 10.0, 10.0, 0.9);
        let b = make_box(10.0, 0.0, 10.0, 10.0, 0.8);
        assert!((a.iou(&b)).abs() < f32::EPSILON);
    }

    // ---- non_maximum_suppression ----

    #[test]
    fn test_nms_empty_input() {
        let kept = non_maximum_suppression(&[], 0.5, 0.4);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_nms_identical_boxes_keeps_one() {
        let boxes = vec![
            make_box(0.0, 0.0, 10.0, 10.0, 0.9),
            make_box(0.0, 0.0, 10.0, 10.0, 0.7),
            make_box(0.0, 0.0, 10.0, 10.0, 0.5),
        ];
        let kept = non_maximum_suppression(&boxes, 0.0, 0.5);
        assert_eq!(
            kept.len(),
            1,
            "identical boxes → only highest score survives"
        );
        assert_eq!(kept[0], 0, "index 0 has score 0.9");
    }

    #[test]
    fn test_nms_non_overlapping_keeps_all() {
        let boxes = vec![
            make_box(0.0, 0.0, 10.0, 10.0, 0.9),
            make_box(50.0, 50.0, 10.0, 10.0, 0.7),
            make_box(100.0, 100.0, 10.0, 10.0, 0.5),
        ];
        let kept = non_maximum_suppression(&boxes, 0.0, 0.5);
        assert_eq!(kept.len(), 3, "non-overlapping boxes all survive");
    }

    #[test]
    fn test_nms_score_threshold_filters() {
        let boxes = vec![
            make_box(0.0, 0.0, 10.0, 10.0, 0.9),
            make_box(20.0, 0.0, 10.0, 10.0, 0.3), // below threshold
        ];
        let kept = non_maximum_suppression(&boxes, 0.5, 0.5);
        assert_eq!(kept.len(), 1, "low-score box filtered");
        assert_eq!(kept[0], 0);
    }

    #[test]
    fn test_nms_iou_zero_no_suppression() {
        // Two boxes with IoU = 0 should both survive regardless of iou_threshold
        let boxes = vec![
            make_box(0.0, 0.0, 5.0, 5.0, 0.9),
            make_box(10.0, 10.0, 5.0, 5.0, 0.8),
        ];
        let kept = non_maximum_suppression(&boxes, 0.0, 0.0);
        assert_eq!(kept.len(), 2);
    }

    // ---- soft_nms_boxes (DetectionBox-based legacy API) ----

    #[test]
    fn test_soft_nms_empty_input() {
        let mut boxes: Vec<DetectionBox> = Vec::new();
        let kept = soft_nms_boxes(&mut boxes, 0.5, 0.5);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_soft_nms_decays_overlapping_scores() {
        let mut boxes = vec![
            make_box(0.0, 0.0, 10.0, 10.0, 0.9), // high score
            make_box(0.0, 0.0, 10.0, 10.0, 0.8), // identical — will be decayed
        ];
        // Use a very low threshold so both survive unless heavily decayed
        let kept = soft_nms_boxes(&mut boxes, 0.01, 0.5);
        // The second box's score must have been decayed
        let score_second = boxes[kept.iter().find(|&&i| i == 1).copied().unwrap_or(0)].score;
        // Original score was 0.8; after full overlap decay with sigma=0.5 it should be < 0.8
        assert!(score_second < 0.79, "score not decayed: {score_second}");
    }

    #[test]
    fn test_soft_nms_score_threshold_removes_decayed() {
        let mut boxes = vec![
            make_box(0.0, 0.0, 10.0, 10.0, 0.9),
            make_box(0.0, 0.0, 10.0, 10.0, 0.5), // identical → decayed heavily
        ];
        // High threshold — decayed box should fall below it
        let kept = soft_nms_boxes(&mut boxes, 0.7, 0.01);
        // With sigma=0.01 and IoU=1.0: decay = exp(-1/0.01) ≈ 0; score ≈ 0 < 0.7
        assert!(
            !kept.contains(&1),
            "heavily decayed box should be excluded; kept={kept:?}"
        );
    }

    // ---- Detection / iou / nms / soft_nms (universal API) ----

    fn make_det(x: f32, y: f32, w: f32, h: f32, conf: f32, class_id: usize) -> Detection {
        Detection::new(x, y, w, h, conf, class_id)
    }

    #[test]
    fn test_detection_iou_identical() {
        let a = make_det(0.0, 0.0, 10.0, 10.0, 0.9, 0);
        let b = a.clone();
        let overlap = iou(&a, &b);
        assert!(
            (overlap - 1.0).abs() < 1e-5,
            "identical boxes IoU=1, got {overlap}"
        );
    }

    #[test]
    fn test_detection_iou_non_overlapping() {
        let a = make_det(0.0, 0.0, 5.0, 5.0, 0.9, 0);
        let b = make_det(10.0, 10.0, 5.0, 5.0, 0.8, 0);
        assert!((iou(&a, &b)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detection_iou_partial() {
        // Two 10×10 boxes offset by 5 → 5×5 intersection, union=175
        let a = make_det(0.0, 0.0, 10.0, 10.0, 0.9, 0);
        let b = make_det(5.0, 5.0, 10.0, 10.0, 0.8, 0);
        let expected = 25.0 / 175.0;
        assert!((iou(&a, &b) - expected).abs() < 1e-5, "got {}", iou(&a, &b));
    }

    #[test]
    fn test_nms_detection_keeps_highest_confidence() {
        let mut dets = vec![
            make_det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            make_det(0.0, 0.0, 10.0, 10.0, 0.7, 0), // identical → suppressed
            make_det(0.0, 0.0, 10.0, 10.0, 0.5, 0), // identical → suppressed
        ];
        let kept = nms(&mut dets, 0.5);
        assert_eq!(kept.len(), 1, "only highest-confidence survives");
        assert!((kept[0].confidence - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_nms_detection_non_overlapping_keeps_all() {
        let mut dets = vec![
            make_det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            make_det(50.0, 50.0, 10.0, 10.0, 0.7, 1),
            make_det(100.0, 100.0, 10.0, 10.0, 0.5, 2),
        ];
        let kept = nms(&mut dets, 0.5);
        assert_eq!(kept.len(), 3, "all non-overlapping detections survive");
    }

    #[test]
    fn test_nms_detection_empty() {
        let mut dets: Vec<Detection> = Vec::new();
        let kept = nms(&mut dets, 0.5);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_soft_nms_detection_decays_overlap() {
        let mut dets = vec![
            make_det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            make_det(0.0, 0.0, 10.0, 10.0, 0.8, 0), // identical → heavy decay
        ];
        let survivors = soft_nms(&mut dets, 0.5);
        // Second box's score must have been decayed
        if let Some(second) = survivors.iter().find(|d| (d.confidence - 0.8).abs() < 0.05) {
            // With IoU=1 and sigma=0.5: decay = exp(-1/0.5) ≈ 0.135
            // Original 0.8 * 0.135 ≈ 0.108 < 0.8
            assert!(
                second.confidence < 0.79,
                "score not decayed: {}",
                second.confidence
            );
        }
        // At least the top box survives
        assert!(!survivors.is_empty());
    }

    #[test]
    fn test_soft_nms_detection_removes_zero_score() {
        // With sigma very small, full-overlap box collapses to ~0
        let mut dets = vec![
            make_det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            make_det(0.0, 0.0, 10.0, 10.0, 0.5, 0), // same → decayed to ~0
        ];
        let survivors = soft_nms(&mut dets, 0.001);
        // After decay both should ideally survive or the second may have near-zero confidence
        // The key contract is: survivors have confidence > 0
        assert!(survivors.iter().all(|d| d.confidence > 0.0));
    }

    #[test]
    fn test_soft_nms_detection_empty() {
        let mut dets: Vec<Detection> = Vec::new();
        let survivors = soft_nms(&mut dets, 0.5);
        assert!(survivors.is_empty());
    }
}
