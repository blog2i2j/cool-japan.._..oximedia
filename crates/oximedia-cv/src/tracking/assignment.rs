//! Hungarian algorithm for assignment problems.
//!
//! Solves the linear assignment problem in O(n³) time, commonly used
//! for track-detection matching in multi-object tracking.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::assignment::hungarian_algorithm;
//!
//! let cost_matrix = vec![
//!     vec![1.0, 2.0, 3.0],
//!     vec![2.0, 4.0, 6.0],
//!     vec![3.0, 6.0, 9.0],
//! ];
//!
//! let assignments = hungarian_algorithm(&cost_matrix);
//! ```

use crate::error::{CvError, CvResult};

/// Solve the linear assignment problem using the Hungarian algorithm.
///
/// Finds the optimal assignment that minimizes the total cost.
///
/// # Arguments
///
/// * `cost_matrix` - Cost matrix where `cost_matrix[i][j]` is the cost of assigning row `i` to column `j`
///
/// # Returns
///
/// Vector of assignments where `assignments[i]` is the column assigned to row `i`,
/// or `None` if row `i` is unassigned.
///
/// # Examples
///
/// ```
/// use oximedia_cv::tracking::assignment::hungarian_algorithm;
///
/// let costs = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![2.0, 4.0, 6.0],
///     vec![3.0, 6.0, 9.0],
/// ];
///
/// let result = hungarian_algorithm(&costs);
/// assert_eq!(result.len(), 3);
/// ```
pub fn hungarian_algorithm(cost_matrix: &[Vec<f64>]) -> Vec<Option<usize>> {
    if cost_matrix.is_empty() {
        return Vec::new();
    }

    let n_rows = cost_matrix.len();
    let n_cols = cost_matrix[0].len();

    // Make square matrix by padding with large values
    let size = n_rows.max(n_cols);
    let mut costs = vec![vec![1e9; size]; size];

    for (i, row) in cost_matrix.iter().enumerate() {
        for (j, &cost) in row.iter().enumerate() {
            costs[i][j] = cost;
        }
    }

    // Run Hungarian algorithm
    let assignments = hungarian_solve(&costs);

    // Extract valid assignments for original matrix
    let mut result = vec![None; n_rows];
    for (i, &j) in assignments.iter().enumerate().take(n_rows) {
        if j < n_cols && costs[i][j] < 1e8 {
            result[i] = Some(j);
        }
    }

    result
}

/// Core Hungarian algorithm implementation.
fn hungarian_solve(cost_matrix: &[Vec<f64>]) -> Vec<usize> {
    let n = cost_matrix.len();
    let mut costs = cost_matrix.to_vec();

    // Step 1: Subtract row minimums
    for row in &mut costs {
        let row_min = row.iter().copied().fold(f64::INFINITY, f64::min);
        for val in row {
            *val -= row_min;
        }
    }

    // Step 2: Subtract column minimums
    for j in 0..n {
        let col_min = (0..n).map(|i| costs[i][j]).fold(f64::INFINITY, f64::min);
        for i in 0..n {
            costs[i][j] -= col_min;
        }
    }

    // Initialize assignments
    let mut row_assigned = vec![false; n];
    let mut col_assigned = vec![false; n];
    let mut assignments = vec![usize::MAX; n];

    // Try to find initial assignments
    for i in 0..n {
        for j in 0..n {
            if costs[i][j].abs() < 1e-10 && !col_assigned[j] {
                assignments[i] = j;
                row_assigned[i] = true;
                col_assigned[j] = true;
                break;
            }
        }
    }

    // Main loop: augment until all rows are assigned
    loop {
        // Find unassigned row
        let unassigned_row = (0..n).find(|&i| !row_assigned[i]);

        let Some(start_row) = unassigned_row else {
            break; // All rows assigned
        };

        // Augment path starting from unassigned row
        if !augment_path(
            &mut costs,
            &mut assignments,
            &mut row_assigned,
            &mut col_assigned,
            start_row,
        ) {
            // No augmenting path found, update costs
            update_costs(&mut costs, &row_assigned, &col_assigned);
        }
    }

    assignments
}

/// Find and augment an augmenting path.
fn augment_path(
    costs: &mut [Vec<f64>],
    assignments: &mut [usize],
    row_assigned: &mut [bool],
    col_assigned: &mut [bool],
    start_row: usize,
) -> bool {
    let n = costs.len();
    let mut visited_rows = vec![false; n];
    let mut visited_cols = vec![false; n];
    let mut parent_row = vec![usize::MAX; n];

    let mut queue = vec![start_row];
    visited_rows[start_row] = true;

    while !queue.is_empty() {
        let mut next_queue = Vec::new();

        for &row in &queue {
            // Find all columns with zero cost from this row
            for col in 0..n {
                if !visited_cols[col] && costs[row][col].abs() < 1e-10 {
                    visited_cols[col] = true;
                    parent_row[col] = row;

                    // Check if this column is unassigned
                    if !col_assigned[col] {
                        // Found augmenting path, update assignments
                        let mut curr_col = col;
                        loop {
                            let curr_row = parent_row[curr_col];
                            let prev_col = assignments[curr_row];

                            assignments[curr_row] = curr_col;
                            row_assigned[curr_row] = true;
                            col_assigned[curr_col] = true;

                            if curr_row == start_row {
                                break;
                            }

                            curr_col = prev_col;
                        }

                        return true;
                    }

                    // Follow assignment to another row (col is assigned, so a row must own it)
                    if let Some(assigned_row) = (0..n).find(|&r| assignments[r] == col) {
                        if !visited_rows[assigned_row] {
                            visited_rows[assigned_row] = true;
                            next_queue.push(assigned_row);
                        }
                    }
                }
            }
        }

        queue = next_queue;
    }

    false
}

/// Update costs to create new zero entries.
fn update_costs(costs: &mut [Vec<f64>], row_assigned: &[bool], col_assigned: &[bool]) {
    let n = costs.len();

    // Find minimum cost in uncovered cells
    let mut min_val = f64::INFINITY;

    for i in 0..n {
        if !row_assigned[i] {
            for j in 0..n {
                if !col_assigned[j] {
                    min_val = min_val.min(costs[i][j]);
                }
            }
        }
    }

    if min_val.is_infinite() {
        return;
    }

    // Subtract minimum from uncovered rows
    for i in 0..n {
        if !row_assigned[i] {
            for j in 0..n {
                costs[i][j] -= min_val;
            }
        }
    }

    // Add minimum to covered columns
    for j in 0..n {
        if col_assigned[j] {
            for i in 0..n {
                costs[i][j] += min_val;
            }
        }
    }
}

/// Compute IoU (Intersection over Union) between two bounding boxes.
///
/// # Examples
///
/// ```
/// use oximedia_cv::tracking::assignment::compute_iou;
/// use oximedia_cv::detect::BoundingBox;
///
/// let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
/// let bbox2 = BoundingBox::new(5.0, 5.0, 10.0, 10.0);
///
/// let iou = compute_iou(&bbox1, &bbox2);
/// assert!(iou > 0.0 && iou < 1.0);
/// ```
pub fn compute_iou(bbox1: &crate::detect::BoundingBox, bbox2: &crate::detect::BoundingBox) -> f64 {
    let x1 = bbox1.x.max(bbox2.x);
    let y1 = bbox1.y.max(bbox2.y);
    let x2 = (bbox1.x + bbox1.width).min(bbox2.x + bbox2.width);
    let y2 = (bbox1.y + bbox1.height).min(bbox2.y + bbox2.height);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = ((x2 - x1) * (y2 - y1)) as f64;
    let area1 = (bbox1.width * bbox1.height) as f64;
    let area2 = (bbox2.width * bbox2.height) as f64;
    let union = area1 + area2 - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Create a cost matrix from IoU distances.
///
/// Cost is defined as 1 - IoU, so lower cost means better match.
///
/// # Arguments
///
/// * `tracks` - Existing tracks (bounding boxes)
/// * `detections` - New detections (bounding boxes)
///
/// # Returns
///
/// Cost matrix where `cost[i][j]` is the cost of matching track `i` to detection `j`.
pub fn create_iou_cost_matrix(
    tracks: &[crate::detect::BoundingBox],
    detections: &[crate::detect::BoundingBox],
) -> Vec<Vec<f64>> {
    let mut costs = vec![vec![0.0; detections.len()]; tracks.len()];

    for (i, track) in tracks.iter().enumerate() {
        for (j, detection) in detections.iter().enumerate() {
            let iou = compute_iou(track, detection);
            costs[i][j] = 1.0 - iou; // Lower cost for higher IoU
        }
    }

    costs
}

/// Filter assignments by maximum cost threshold.
///
/// # Arguments
///
/// * `assignments` - Assignments from Hungarian algorithm
/// * `cost_matrix` - Original cost matrix
/// * `max_cost` - Maximum allowed cost
///
/// # Returns
///
/// Filtered assignments where high-cost matches are set to `None`.
pub fn filter_assignments_by_cost(
    assignments: &[Option<usize>],
    cost_matrix: &[Vec<f64>],
    max_cost: f64,
) -> Vec<Option<usize>> {
    let mut filtered = assignments.to_vec();

    for (i, &assignment) in assignments.iter().enumerate() {
        if let Some(j) = assignment {
            if i < cost_matrix.len() && j < cost_matrix[i].len() && cost_matrix[i][j] > max_cost {
                filtered[i] = None;
            }
        }
    }

    filtered
}

/// Greedy assignment based on cost (simpler alternative to Hungarian).
///
/// Assigns items greedily in order of increasing cost.
///
/// # Arguments
///
/// * `cost_matrix` - Cost matrix
/// * `max_cost` - Maximum allowed cost
///
/// # Returns
///
/// Vector of assignments.
pub fn greedy_assignment(cost_matrix: &[Vec<f64>], max_cost: f64) -> Vec<Option<usize>> {
    if cost_matrix.is_empty() {
        return Vec::new();
    }

    let n_rows = cost_matrix.len();
    let n_cols = cost_matrix[0].len();

    // Create list of (cost, row, col) tuples
    let mut costs = Vec::new();
    for (i, row) in cost_matrix.iter().enumerate() {
        for (j, &cost) in row.iter().enumerate() {
            if cost <= max_cost {
                costs.push((cost, i, j));
            }
        }
    }

    // Sort by cost
    costs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign greedily
    let mut assignments = vec![None; n_rows];
    let mut col_taken = vec![false; n_cols];

    for (_, i, j) in costs {
        if assignments[i].is_none() && !col_taken[j] {
            assignments[i] = Some(j);
            col_taken[j] = true;
        }
    }

    assignments
}
