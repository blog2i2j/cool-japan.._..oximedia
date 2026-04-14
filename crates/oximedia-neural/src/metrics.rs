//! Classification metrics: confusion matrix, precision, recall, F1-score,
//! accuracy, and per-class reports.
//!
//! All computation is done entirely in pure Rust without external dependencies.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::metrics::ConfusionMatrix;
//!
//! let mut cm = ConfusionMatrix::new(3);
//! cm.update(0, 0); // true positive for class 0
//! cm.update(1, 2); // class 1 predicted as class 2 (false negative for 1)
//! cm.update(2, 2); // true positive for class 2
//!
//! let report = cm.classification_report();
//! assert!(report.macro_f1 >= 0.0 && report.macro_f1 <= 1.0);
//! ```

/// Confusion matrix for multi-class classification.
///
/// Rows are **true** classes; columns are **predicted** classes.
/// `matrix[true_class][predicted_class] += 1` for each sample.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Flat row-major buffer: `matrix[i * n_classes + j]`.
    matrix: Vec<u64>,
    /// Number of classes.
    n_classes: usize,
    /// Total number of samples seen.
    total_samples: u64,
}

impl ConfusionMatrix {
    /// Create a new zero-filled confusion matrix for `n_classes` classes.
    ///
    /// # Panics
    ///
    /// Does not panic; `n_classes = 0` simply creates an empty matrix.
    #[must_use]
    pub fn new(n_classes: usize) -> Self {
        Self {
            matrix: vec![0; n_classes * n_classes],
            n_classes,
            total_samples: 0,
        }
    }

    /// Increment the cell at `(true_class, predicted_class)`.
    ///
    /// Silently ignores out-of-range indices.
    pub fn update(&mut self, true_class: usize, predicted_class: usize) {
        if true_class < self.n_classes && predicted_class < self.n_classes {
            self.matrix[true_class * self.n_classes + predicted_class] += 1;
            self.total_samples += 1;
        }
    }

    /// Batch update from parallel slices of true and predicted labels.
    ///
    /// `true_labels` and `predicted_labels` must have the same length.
    pub fn update_batch(&mut self, true_labels: &[usize], predicted_labels: &[usize]) {
        for (&t, &p) in true_labels.iter().zip(predicted_labels.iter()) {
            self.update(t, p);
        }
    }

    /// Reset all counts to zero.
    pub fn reset(&mut self) {
        for v in &mut self.matrix {
            *v = 0;
        }
        self.total_samples = 0;
    }

    /// Number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Total number of samples seen.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Read a single cell: `(true_class, predicted_class)`.
    #[must_use]
    pub fn get(&self, true_class: usize, predicted_class: usize) -> u64 {
        if true_class < self.n_classes && predicted_class < self.n_classes {
            self.matrix[true_class * self.n_classes + predicted_class]
        } else {
            0
        }
    }

    /// Overall accuracy: fraction of correctly classified samples.
    #[must_use]
    pub fn accuracy(&self) -> f32 {
        if self.total_samples == 0 {
            return 0.0;
        }
        let correct: u64 = (0..self.n_classes).map(|i| self.get(i, i)).sum();
        correct as f32 / self.total_samples as f32
    }

    /// True-positive count for a single class.
    #[must_use]
    pub fn true_positives(&self, class: usize) -> u64 {
        self.get(class, class)
    }

    /// False-positive count: predicted as `class` but actually another class.
    #[must_use]
    pub fn false_positives(&self, class: usize) -> u64 {
        if class >= self.n_classes {
            return 0;
        }
        (0..self.n_classes)
            .filter(|&r| r != class)
            .map(|r| self.get(r, class))
            .sum()
    }

    /// False-negative count: actually `class` but predicted as another class.
    #[must_use]
    pub fn false_negatives(&self, class: usize) -> u64 {
        if class >= self.n_classes {
            return 0;
        }
        (0..self.n_classes)
            .filter(|&c| c != class)
            .map(|c| self.get(class, c))
            .sum()
    }

    /// Precision for a class: `TP / (TP + FP)`.  Returns 0 when denominator is 0.
    #[must_use]
    pub fn precision(&self, class: usize) -> f32 {
        let tp = self.true_positives(class);
        let fp = self.false_positives(class);
        let denom = tp + fp;
        if denom == 0 {
            0.0
        } else {
            tp as f32 / denom as f32
        }
    }

    /// Recall (sensitivity) for a class: `TP / (TP + FN)`.
    /// Returns 0 when denominator is 0.
    #[must_use]
    pub fn recall(&self, class: usize) -> f32 {
        let tp = self.true_positives(class);
        let fn_ = self.false_negatives(class);
        let denom = tp + fn_;
        if denom == 0 {
            0.0
        } else {
            tp as f32 / denom as f32
        }
    }

    /// F1-score for a class: `2 * precision * recall / (precision + recall)`.
    /// Returns 0 when both precision and recall are 0.
    #[must_use]
    pub fn f1_score(&self, class: usize) -> f32 {
        let p = self.precision(class);
        let r = self.recall(class);
        let denom = p + r;
        if denom < f32::EPSILON {
            0.0
        } else {
            2.0 * p * r / denom
        }
    }

    /// Fbeta-score for a class: `(1 + β²) * P * R / (β² * P + R)`.
    #[must_use]
    pub fn fbeta_score(&self, class: usize, beta: f32) -> f32 {
        let p = self.precision(class);
        let r = self.recall(class);
        let beta_sq = beta * beta;
        let denom = beta_sq * p + r;
        if denom < f32::EPSILON {
            0.0
        } else {
            (1.0 + beta_sq) * p * r / denom
        }
    }

    /// F1-score for every class as a vector.
    ///
    /// Returns a `Vec<f32>` of length `n_classes` where element `i` is the
    /// F1-score for class `i`. Returns an empty vector when `n_classes == 0`.
    #[must_use]
    pub fn f1_scores(&self) -> Vec<f32> {
        (0..self.n_classes).map(|c| self.f1_score(c)).collect()
    }

    /// Macro-averaged F1-score across all classes.
    #[must_use]
    pub fn macro_f1(&self) -> f32 {
        if self.n_classes == 0 {
            return 0.0;
        }
        let sum: f32 = (0..self.n_classes).map(|c| self.f1_score(c)).sum();
        sum / self.n_classes as f32
    }

    /// Weighted F1-score: weighted average by the number of true samples per class.
    #[must_use]
    pub fn weighted_f1(&self) -> f32 {
        if self.total_samples == 0 {
            return 0.0;
        }
        let weighted_sum: f32 = (0..self.n_classes)
            .map(|c| {
                let support: u64 = (0..self.n_classes).map(|j| self.get(c, j)).sum();
                self.f1_score(c) * support as f32
            })
            .sum();
        weighted_sum / self.total_samples as f32
    }

    /// Macro-averaged precision.
    #[must_use]
    pub fn macro_precision(&self) -> f32 {
        if self.n_classes == 0 {
            return 0.0;
        }
        (0..self.n_classes).map(|c| self.precision(c)).sum::<f32>() / self.n_classes as f32
    }

    /// Macro-averaged recall.
    #[must_use]
    pub fn macro_recall(&self) -> f32 {
        if self.n_classes == 0 {
            return 0.0;
        }
        (0..self.n_classes).map(|c| self.recall(c)).sum::<f32>() / self.n_classes as f32
    }

    /// Per-class metrics.
    #[must_use]
    pub fn per_class_metrics(&self) -> Vec<ClassMetrics> {
        (0..self.n_classes)
            .map(|c| {
                let support: u64 = (0..self.n_classes).map(|j| self.get(c, j)).sum();
                ClassMetrics {
                    class_id: c,
                    precision: self.precision(c),
                    recall: self.recall(c),
                    f1: self.f1_score(c),
                    support,
                }
            })
            .collect()
    }

    /// Full classification report.
    #[must_use]
    pub fn classification_report(&self) -> ClassificationReport {
        ClassificationReport {
            accuracy: self.accuracy(),
            macro_f1: self.macro_f1(),
            weighted_f1: self.weighted_f1(),
            macro_precision: self.macro_precision(),
            macro_recall: self.macro_recall(),
            per_class: self.per_class_metrics(),
            total_samples: self.total_samples,
        }
    }
}

/// Metrics for a single class.
#[derive(Debug, Clone)]
pub struct ClassMetrics {
    /// Class identifier.
    pub class_id: usize,
    /// Precision.
    pub precision: f32,
    /// Recall.
    pub recall: f32,
    /// F1-score.
    pub f1: f32,
    /// Number of true samples (support).
    pub support: u64,
}

/// Aggregated classification report.
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    /// Overall accuracy.
    pub accuracy: f32,
    /// Macro-averaged F1.
    pub macro_f1: f32,
    /// Weighted F1.
    pub weighted_f1: f32,
    /// Macro-averaged precision.
    pub macro_precision: f32,
    /// Macro-averaged recall.
    pub macro_recall: f32,
    /// Per-class metrics.
    pub per_class: Vec<ClassMetrics>,
    /// Total samples.
    pub total_samples: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Perfect predictions for 3 classes.
    fn perfect_cm() -> ConfusionMatrix {
        let mut cm = ConfusionMatrix::new(3);
        for _ in 0..5 {
            cm.update(0, 0);
        }
        for _ in 0..5 {
            cm.update(1, 1);
        }
        for _ in 0..5 {
            cm.update(2, 2);
        }
        cm
    }

    #[test]
    fn test_accuracy_perfect() {
        let cm = perfect_cm();
        assert!((cm.accuracy() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_accuracy_zero_samples() {
        let cm = ConfusionMatrix::new(3);
        assert!((cm.accuracy() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_precision_recall_perfect() {
        let cm = perfect_cm();
        for c in 0..3 {
            assert!((cm.precision(c) - 1.0).abs() < 1e-5, "precision class {c}");
            assert!((cm.recall(c) - 1.0).abs() < 1e-5, "recall class {c}");
            assert!((cm.f1_score(c) - 1.0).abs() < 1e-5, "f1 class {c}");
        }
    }

    #[test]
    fn test_macro_f1_perfect() {
        let cm = perfect_cm();
        assert!((cm.macro_f1() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_binary_classification() {
        // Binary: TP=4, FP=1, FN=1, TN=4
        let mut cm = ConfusionMatrix::new(2);
        // True class 0, predicted 0 (TP) ×4
        for _ in 0..4 {
            cm.update(0, 0);
        }
        // True class 0, predicted 1 (FN for 0) ×1
        cm.update(0, 1);
        // True class 1, predicted 0 (FP for 0) ×1
        cm.update(1, 0);
        // True class 1, predicted 1 (TN) ×4
        for _ in 0..4 {
            cm.update(1, 1);
        }

        let p = cm.precision(0);
        let r = cm.recall(0);
        let f1 = cm.f1_score(0);

        // precision(0) = TP/(TP+FP) = 4/5 = 0.8
        assert!((p - 0.8).abs() < 1e-5, "precision={p}");
        // recall(0) = TP/(TP+FN) = 4/5 = 0.8
        assert!((r - 0.8).abs() < 1e-5, "recall={r}");
        // F1 = 2*0.8*0.8/(0.8+0.8) = 0.8
        assert!((f1 - 0.8).abs() < 1e-5, "f1={f1}");
    }

    #[test]
    fn test_true_positives_false_positives_false_negatives() {
        let mut cm = ConfusionMatrix::new(3);
        cm.update(0, 0); // TP class 0
        cm.update(0, 1); // FN class 0, FP class 1
        cm.update(1, 2); // FN class 1, FP class 2
        cm.update(2, 2); // TP class 2
        cm.update(2, 0); // FN class 2, FP class 0

        assert_eq!(cm.true_positives(0), 1);
        assert_eq!(cm.false_positives(0), 1);
        assert_eq!(cm.false_negatives(0), 1);

        assert_eq!(cm.true_positives(1), 0);
        assert_eq!(cm.true_positives(2), 1);
    }

    #[test]
    fn test_update_batch() {
        let mut cm = ConfusionMatrix::new(2);
        let true_labels = vec![0, 0, 1, 1];
        let pred_labels = vec![0, 1, 0, 1];
        cm.update_batch(&true_labels, &pred_labels);
        assert_eq!(cm.total_samples(), 4);
        assert_eq!(cm.true_positives(0), 1);
        assert_eq!(cm.true_positives(1), 1);
    }

    #[test]
    fn test_reset() {
        let mut cm = perfect_cm();
        assert!(cm.total_samples() > 0);
        cm.reset();
        assert_eq!(cm.total_samples(), 0);
        assert!((cm.accuracy()).abs() < 1e-5);
    }

    #[test]
    fn test_classification_report_fields_bounded() {
        let cm = perfect_cm();
        let report = cm.classification_report();
        assert!(report.accuracy >= 0.0 && report.accuracy <= 1.0);
        assert!(report.macro_f1 >= 0.0 && report.macro_f1 <= 1.0);
        assert!(report.weighted_f1 >= 0.0 && report.weighted_f1 <= 1.0);
        assert_eq!(report.per_class.len(), 3);
        assert_eq!(report.total_samples, 15);
    }

    #[test]
    fn test_fbeta_score_f2() {
        let mut cm = ConfusionMatrix::new(2);
        for _ in 0..4 {
            cm.update(0, 0);
        }
        cm.update(0, 1);
        cm.update(1, 0);
        for _ in 0..4 {
            cm.update(1, 1);
        }

        let f2 = cm.fbeta_score(0, 2.0);
        let p = cm.precision(0); // 0.8
        let r = cm.recall(0); // 0.8
        let expected = 5.0 * p * r / (4.0 * p + r);
        assert!((f2 - expected).abs() < 1e-4, "f2={f2} expected={expected}");
    }

    #[test]
    fn test_zero_division_precision_recall() {
        // Class with no predictions and no samples
        let cm = ConfusionMatrix::new(3);
        assert!((cm.precision(0)).abs() < 1e-5);
        assert!((cm.recall(0)).abs() < 1e-5);
        assert!((cm.f1_score(0)).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_f1_matches_macro_on_balanced() {
        // With equal support for all classes, weighted == macro
        let cm = perfect_cm();
        let wf1 = cm.weighted_f1();
        let mf1 = cm.macro_f1();
        assert!((wf1 - mf1).abs() < 1e-5, "wf1={wf1} mf1={mf1}");
    }

    #[test]
    fn test_out_of_range_class_ignored() {
        let mut cm = ConfusionMatrix::new(2);
        cm.update(5, 0); // out of range, ignored
        cm.update(0, 5); // out of range, ignored
        assert_eq!(cm.total_samples(), 0);
    }
}
