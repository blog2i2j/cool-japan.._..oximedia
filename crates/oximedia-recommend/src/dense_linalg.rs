//! Pure Rust dense linear algebra types replacing ndarray dependency.
//!
//! Provides `DenseMatrix` (2D: rows x cols) and `DenseVector` (1D) for
//! recommendation algorithms without external linear algebra crates.

use serde::{Deserialize, Serialize};

/// A 2-dimensional dense matrix backed by a flat `Vec<f32>`.
///
/// This replaces `ndarray::Array2<f32>` for storing factor matrices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseMatrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl DenseMatrix {
    /// Create a new matrix filled with zeros.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0f32; rows * cols],
            rows,
            cols,
        }
    }

    /// Get a value at (row, col).
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Set a value at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    /// Get the number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Get a row as a `Vec<f32>`.
    #[must_use]
    pub fn row_vec(&self, row: usize) -> Vec<f32> {
        let start = row * self.cols;
        self.data[start..start + self.cols].to_vec()
    }

    /// Get a row as a slice.
    #[must_use]
    pub fn row_slice(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Get a column as a `Vec<f32>`.
    #[must_use]
    pub fn col_vec(&self, col: usize) -> Vec<f32> {
        (0..self.rows)
            .map(|r| self.data[r * self.cols + col])
            .collect()
    }

    /// Concatenate rows (add rows from another matrix with same number of columns).
    #[must_use]
    pub fn concat_rows(&self, other: &Self) -> Self {
        debug_assert_eq!(self.cols, other.cols);
        let mut new_data = self.data.clone();
        new_data.extend_from_slice(&other.data);
        Self {
            data: new_data,
            rows: self.rows + other.rows,
            cols: self.cols,
        }
    }

    /// Concatenate columns (add columns from another matrix with same number of rows).
    #[must_use]
    pub fn concat_cols(&self, other: &Self) -> Self {
        debug_assert_eq!(self.rows, other.rows);
        let new_cols = self.cols + other.cols;
        let mut new_data = vec![0.0f32; self.rows * new_cols];

        for r in 0..self.rows {
            // Copy self row
            let dst_start = r * new_cols;
            let src_start = r * self.cols;
            new_data[dst_start..dst_start + self.cols]
                .copy_from_slice(&self.data[src_start..src_start + self.cols]);
            // Copy other row
            let other_src_start = r * other.cols;
            new_data[dst_start + self.cols..dst_start + new_cols]
                .copy_from_slice(&other.data[other_src_start..other_src_start + other.cols]);
        }

        Self {
            data: new_data,
            rows: self.rows,
            cols: new_cols,
        }
    }
}

impl PartialEq for DenseMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

/// A 1-dimensional dense vector backed by a `Vec<f32>`.
///
/// This replaces `ndarray::Array1<f32>`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenseVector {
    data: Vec<f32>,
}

impl DenseVector {
    /// Create a new vector from a `Vec<f32>`.
    #[must_use]
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Create a vector of zeros.
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0f32; len],
        }
    }

    /// Get the length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a value at index.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> f32 {
        self.data[index]
    }

    /// Set a value at index.
    #[inline]
    pub fn set(&mut self, index: usize, value: f32) {
        self.data[index] = value;
    }

    /// Get the underlying data as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Convert to a `Vec<f32>`.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Iterate over the values.
    pub fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.data.iter()
    }

    /// Compute dot product with another vector.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_matrix_zeros() {
        let m = DenseMatrix::zeros(3, 4);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 4);
        assert!((m.get(0, 0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dense_matrix_set_get() {
        let mut m = DenseMatrix::zeros(3, 3);
        m.set(1, 2, 5.0);
        assert!((m.get(1, 2) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dense_matrix_row_vec() {
        let mut m = DenseMatrix::zeros(2, 3);
        m.set(1, 0, 1.0);
        m.set(1, 1, 2.0);
        m.set(1, 2, 3.0);
        assert_eq!(m.row_vec(1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dense_matrix_col_vec() {
        let mut m = DenseMatrix::zeros(3, 2);
        m.set(0, 1, 10.0);
        m.set(1, 1, 20.0);
        m.set(2, 1, 30.0);
        assert_eq!(m.col_vec(1), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_dense_matrix_concat_rows() {
        let m1 = DenseMatrix::zeros(2, 3);
        let m2 = DenseMatrix::zeros(3, 3);
        let result = m1.concat_rows(&m2);
        assert_eq!(result.nrows(), 5);
        assert_eq!(result.ncols(), 3);
    }

    #[test]
    fn test_dense_matrix_concat_cols() {
        let m1 = DenseMatrix::zeros(2, 3);
        let m2 = DenseMatrix::zeros(2, 4);
        let result = m1.concat_cols(&m2);
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 7);
    }

    #[test]
    fn test_dense_vector_from_vec() {
        let v = DenseVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert!((v.get(1) - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dense_vector_dot() {
        let v1 = DenseVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = DenseVector::from_vec(vec![4.0, 5.0, 6.0]);
        assert!((v1.dot(&v2) - 32.0).abs() < f32::EPSILON);
    }
}
