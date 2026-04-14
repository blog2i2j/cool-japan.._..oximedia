//! Strided tensor views with pre-computed stride arrays for fast flat indexing.
//!
//! The core [`Tensor`] type stores shape metadata but
//! computes strides on-the-fly via a reverse loop each time an element is
//! accessed.  This module provides [`StridedView`], a zero-copy view over a
//! `Tensor` that pre-computes and caches the stride array once at creation,
//! allowing O(rank) element lookups via a single dot-product instead of the
//! reverse-accumulation loop.
//!
//! Additionally, [`StridedView`] supports **non-contiguous views** created by
//! slicing or transposing, where strides may differ from the dense row-major
//! layout.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::strided_tensor::StridedView;
//! use oximedia_neural::tensor::Tensor;
//!
//! let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//! let view = StridedView::from_tensor(&t);
//!
//! assert_eq!(view.get(&[0, 2]).unwrap(), 3.0);
//! assert_eq!(view.get(&[1, 0]).unwrap(), 4.0);
//! assert_eq!(view.strides(), &[3, 1]);
//! ```

#![allow(dead_code)]

use crate::error::NeuralError;
use crate::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Stride computation
// ─────────────────────────────────────────────────────────────────────────────

/// Computes row-major (C-contiguous) strides for a given shape.
///
/// For shape `[s0, s1, …, s_{n-1}]` the strides are:
/// ```text
/// stride[i] = Π_{j=i+1}^{n-1} s_j
/// ```
/// so `stride[n-1] = 1`.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let n = shape.len();
    let mut strides = vec![1_usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Converts a flat index to multi-dimensional indices given pre-computed strides.
///
/// This is the inverse of the flat-index calculation:
/// `indices[i] = (flat / strides[i]) % shape[i]`.
pub fn unravel_index(flat: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut indices = Vec::with_capacity(shape.len());
    let mut remaining = flat;
    for (i, &stride) in strides.iter().enumerate() {
        let idx = remaining.checked_div(stride).unwrap_or(0);
        indices.push(idx % shape[i]);
        remaining = remaining.checked_rem(stride).unwrap_or(remaining);
    }
    indices
}

/// Converts multi-dimensional indices to a flat index via dot product with strides.
///
/// No bounds checking — the caller is responsible for validity.
#[inline]
pub fn ravel_index(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides.iter())
        .map(|(&i, &s)| i * s)
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// StridedView
// ─────────────────────────────────────────────────────────────────────────────

/// A read-only view into a flat data buffer with pre-computed strides.
///
/// Supports both contiguous (row-major) and non-contiguous (transposed,
/// sliced) layouts.  Does **not** own the data.
#[derive(Debug, Clone)]
pub struct StridedView<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl<'a> StridedView<'a> {
    /// Creates a contiguous row-major view over a `Tensor`.
    pub fn from_tensor(tensor: &'a Tensor) -> Self {
        let strides = compute_strides(tensor.shape());
        Self {
            data: tensor.data(),
            shape: tensor.shape().to_vec(),
            strides,
            offset: 0,
        }
    }

    /// Creates a view with custom strides and offset.
    ///
    /// This is the low-level constructor used by `transpose` and `slice`.
    pub fn new(
        data: &'a [f32],
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Result<Self, NeuralError> {
        if shape.len() != strides.len() {
            return Err(NeuralError::InvalidShape(format!(
                "StridedView: shape rank {} != strides rank {}",
                shape.len(),
                strides.len()
            )));
        }
        Ok(Self {
            data,
            shape,
            strides,
            offset,
        })
    }

    /// Returns the shape of this view.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the strides of this view.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of logical elements in the view.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns the element at the given multi-dimensional indices.
    pub fn get(&self, indices: &[usize]) -> Result<f32, NeuralError> {
        if indices.len() != self.shape.len() {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "StridedView: expected {} indices, got {}",
                self.shape.len(),
                indices.len()
            )));
        }
        for (i, (&idx, &size)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= size {
                return Err(NeuralError::IndexOutOfBounds(format!(
                    "StridedView: index {} out of bounds for dim {} of size {}",
                    idx, i, size
                )));
            }
        }
        let flat = self.offset + ravel_index(indices, &self.strides);
        if flat >= self.data.len() {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "StridedView: flat index {} out of bounds (buffer len {})",
                flat,
                self.data.len()
            )));
        }
        Ok(self.data[flat])
    }

    /// Creates a transposed view by swapping two axes.
    ///
    /// No data is copied; only the shape and strides are permuted.
    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self, NeuralError> {
        if axis_a >= self.ndim() || axis_b >= self.ndim() {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "StridedView::transpose: axes ({}, {}) out of range for rank {}",
                axis_a,
                axis_b,
                self.ndim()
            )));
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(axis_a, axis_b);
        new_strides.swap(axis_a, axis_b);
        Ok(Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Creates a sub-view by slicing a single index along a given axis.
    ///
    /// Reduces the rank by one (the sliced dimension is removed).
    pub fn select(&self, axis: usize, index: usize) -> Result<Self, NeuralError> {
        if axis >= self.ndim() {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "StridedView::select: axis {} >= rank {}",
                axis,
                self.ndim()
            )));
        }
        if index >= self.shape[axis] {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "StridedView::select: index {} >= dim size {} on axis {}",
                index, self.shape[axis], axis
            )));
        }
        let new_offset = self.offset + index * self.strides[axis];
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(axis);
        new_strides.remove(axis);
        Ok(Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
        })
    }

    /// Materialises this (possibly non-contiguous) view into a new owned `Tensor`
    /// with contiguous row-major layout.
    pub fn to_tensor(&self) -> Result<Tensor, NeuralError> {
        let numel = self.numel();
        if numel == 0 {
            return Err(NeuralError::InvalidShape(
                "StridedView::to_tensor: cannot materialise empty view".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(numel);
        let dense_strides = compute_strides(&self.shape);

        for flat in 0..numel {
            let indices = unravel_index(flat, &self.shape, &dense_strides);
            let src = self.offset + ravel_index(&indices, &self.strides);
            if src >= self.data.len() {
                return Err(NeuralError::IndexOutOfBounds(format!(
                    "StridedView::to_tensor: source index {} >= buffer len {}",
                    src,
                    self.data.len()
                )));
            }
            out.push(self.data[src]);
        }

        Tensor::from_data(out, self.shape.clone())
    }

    /// Returns `true` if this view is contiguous (dense, row-major).
    pub fn is_contiguous(&self) -> bool {
        let expected = compute_strides(&self.shape);
        self.strides == expected && self.offset == 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OwnedStridedTensor
// ─────────────────────────────────────────────────────────────────────────────

/// An owned, strided tensor that carries its data buffer internally.
///
/// Unlike [`StridedView`], this type owns the data and can be moved and stored
/// without lifetime concerns.  Useful for the result of operations like
/// `transpose().to_owned_strided()`.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedStridedTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl OwnedStridedTensor {
    /// Creates a contiguous owned strided tensor from a `Tensor`.
    pub fn from_tensor(tensor: Tensor) -> Self {
        let strides = compute_strides(tensor.shape());
        let shape = tensor.shape().to_vec();
        Self {
            data: tensor.data.clone(),
            shape,
            strides,
            offset: 0,
        }
    }

    /// Returns the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the element at the given indices.
    pub fn get(&self, indices: &[usize]) -> Result<f32, NeuralError> {
        if indices.len() != self.shape.len() {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "OwnedStridedTensor: expected {} indices, got {}",
                self.shape.len(),
                indices.len()
            )));
        }
        for (i, (&idx, &size)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= size {
                return Err(NeuralError::IndexOutOfBounds(format!(
                    "OwnedStridedTensor: index {} out of bounds for dim {} of size {}",
                    idx, i, size
                )));
            }
        }
        let flat = self.offset + ravel_index(indices, &self.strides);
        if flat >= self.data.len() {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "OwnedStridedTensor: flat index {} out of bounds",
                flat
            )));
        }
        Ok(self.data[flat])
    }

    /// Creates a borrow-based `StridedView` from this owned tensor.
    pub fn view(&self) -> StridedView<'_> {
        StridedView {
            data: &self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }

    /// Materialises into a contiguous `Tensor`.
    pub fn to_tensor(&self) -> Result<Tensor, NeuralError> {
        self.view().to_tensor()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // ── compute_strides ──────────────────────────────────────────────────────

    #[test]
    fn test_strides_1d() {
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_strides_2d() {
        assert_eq!(compute_strides(&[3, 4]), vec![4, 1]);
    }

    #[test]
    fn test_strides_3d() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn test_strides_empty() {
        let s: Vec<usize> = compute_strides(&[]);
        assert!(s.is_empty());
    }

    // ── ravel / unravel ──────────────────────────────────────────────────────

    #[test]
    fn test_ravel_unravel_roundtrip() {
        let shape = vec![2, 3, 4];
        let strides = compute_strides(&shape);
        for flat in 0..24 {
            let indices = unravel_index(flat, &shape, &strides);
            let recovered = ravel_index(&indices, &strides);
            assert_eq!(recovered, flat, "flat={}, indices={:?}", flat, indices);
        }
    }

    // ── StridedView ──────────────────────────────────────────────────────────

    #[test]
    fn test_view_get_2d() {
        let t =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("from_data");
        let view = StridedView::from_tensor(&t);

        assert_eq!(view.get(&[0, 0]).expect("get"), 1.0);
        assert_eq!(view.get(&[0, 2]).expect("get"), 3.0);
        assert_eq!(view.get(&[1, 0]).expect("get"), 4.0);
        assert_eq!(view.get(&[1, 2]).expect("get"), 6.0);
    }

    #[test]
    fn test_view_out_of_bounds() {
        let t = Tensor::from_data(vec![1.0, 2.0], vec![2]).expect("from_data");
        let view = StridedView::from_tensor(&t);
        assert!(view.get(&[2]).is_err());
    }

    #[test]
    fn test_view_wrong_rank() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("from_data");
        let view = StridedView::from_tensor(&t);
        assert!(view.get(&[0]).is_err());
    }

    #[test]
    fn test_view_transpose() {
        let t =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("from_data");
        let view = StridedView::from_tensor(&t);
        let tv = view.transpose(0, 1).expect("transpose");

        assert_eq!(tv.shape(), &[3, 2]);
        assert_eq!(tv.strides(), &[1, 3]);
        // Transposed access: row 1, col 0 in the original is [1, 0] = 4.0.
        // In the transposed view that becomes [0, 1].
        assert_eq!(tv.get(&[0, 1]).expect("get"), 4.0);
        assert_eq!(tv.get(&[2, 0]).expect("get"), 3.0);
    }

    #[test]
    fn test_view_select() {
        // [2, 3] tensor; select axis=0, index=1 → [3] tensor with [4, 5, 6].
        let t =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("from_data");
        let view = StridedView::from_tensor(&t);
        let row1 = view.select(0, 1).expect("select");

        assert_eq!(row1.shape(), &[3]);
        assert_eq!(row1.get(&[0]).expect("get"), 4.0);
        assert_eq!(row1.get(&[1]).expect("get"), 5.0);
        assert_eq!(row1.get(&[2]).expect("get"), 6.0);
    }

    #[test]
    fn test_view_to_tensor() {
        let t =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("from_data");
        let view = StridedView::from_tensor(&t);
        let tv = view.transpose(0, 1).expect("transpose");
        let materialized = tv.to_tensor().expect("to_tensor");

        assert_eq!(materialized.shape(), &[3, 2]);
        // Row 0 of transposed: [1.0, 4.0]
        assert_eq!(materialized.get(&[0, 0]).expect("get"), 1.0);
        assert_eq!(materialized.get(&[0, 1]).expect("get"), 4.0);
    }

    #[test]
    fn test_is_contiguous() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("from_data");
        let view = StridedView::from_tensor(&t);
        assert!(view.is_contiguous());

        let tv = view.transpose(0, 1).expect("transpose");
        assert!(!tv.is_contiguous());
    }

    // ── OwnedStridedTensor ───────────────────────────────────────────────────

    #[test]
    fn test_owned_from_tensor_get() {
        let t = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3]).expect("from_data");
        let owned = OwnedStridedTensor::from_tensor(t);
        assert_eq!(owned.get(&[1]).expect("get"), 20.0);
    }

    #[test]
    fn test_owned_to_tensor() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("from_data");
        let owned = OwnedStridedTensor::from_tensor(t.clone());
        let recovered = owned.to_tensor().expect("to_tensor");
        assert_eq!(recovered.data(), t.data());
    }

    #[test]
    fn test_owned_view() {
        let t =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("from_data");
        let owned = OwnedStridedTensor::from_tensor(t);
        let view = owned.view();
        assert_eq!(view.get(&[1, 2]).expect("get"), 6.0);
    }

    #[test]
    fn test_select_axis_out_of_bounds() {
        let t = Tensor::from_data(vec![1.0, 2.0], vec![2]).expect("from_data");
        let view = StridedView::from_tensor(&t);
        assert!(view.select(1, 0).is_err());
    }

    #[test]
    fn test_transpose_axis_out_of_bounds() {
        let t = Tensor::from_data(vec![1.0, 2.0], vec![2]).expect("from_data");
        let view = StridedView::from_tensor(&t);
        assert!(view.transpose(0, 1).is_err());
    }
}
