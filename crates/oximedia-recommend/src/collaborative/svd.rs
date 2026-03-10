//! Singular Value Decomposition for collaborative filtering.

use crate::dense_linalg::{DenseMatrix, DenseVector};
use crate::error::{RecommendError, RecommendResult};
use serde::{Deserialize, Serialize};

/// SVD model for matrix factorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvdModel {
    /// User factor matrix (users x factors)
    user_factors: DenseMatrix,
    /// Item factor matrix (items x factors)
    item_factors: DenseMatrix,
    /// Number of latent factors
    num_factors: usize,
    /// Global mean rating
    global_mean: f32,
}

impl SvdModel {
    /// Create a new SVD model
    #[must_use]
    pub fn new(num_users: usize, num_items: usize, num_factors: usize) -> Self {
        Self {
            user_factors: DenseMatrix::zeros(num_users, num_factors),
            item_factors: DenseMatrix::zeros(num_items, num_factors),
            num_factors,
            global_mean: 0.0,
        }
    }

    /// Train the model using gradient descent
    ///
    /// # Errors
    ///
    /// Returns an error if training fails
    pub fn train(
        &mut self,
        ratings: &[(usize, usize, f32)],
        epochs: usize,
        learning_rate: f32,
        regularization: f32,
    ) -> RecommendResult<()> {
        // Calculate global mean
        if ratings.is_empty() {
            return Err(RecommendError::insufficient_data(
                "No ratings provided for training",
            ));
        }

        self.global_mean = ratings.iter().map(|(_, _, r)| r).sum::<f32>() / ratings.len() as f32;

        // Initialize factors
        self.initialize_factors();

        // Gradient descent
        for _ in 0..epochs {
            for &(user_idx, item_idx, rating) in ratings {
                let prediction = self.predict_internal(user_idx, item_idx);
                let error = rating - prediction;

                // Update factors
                for f in 0..self.num_factors {
                    let user_factor = self.user_factors.get(user_idx, f);
                    let item_factor = self.item_factors.get(item_idx, f);

                    self.user_factors.set(
                        user_idx,
                        f,
                        user_factor
                            + learning_rate * (error * item_factor - regularization * user_factor),
                    );
                    self.item_factors.set(
                        item_idx,
                        f,
                        item_factor
                            + learning_rate * (error * user_factor - regularization * item_factor),
                    );
                }
            }
        }

        Ok(())
    }

    /// Predict rating for user-item pair (internal indices)
    fn predict_internal(&self, user_idx: usize, item_idx: usize) -> f32 {
        if user_idx >= self.user_factors.nrows() || item_idx >= self.item_factors.nrows() {
            return self.global_mean;
        }

        let user_row = self.user_factors.row_slice(user_idx);
        let item_row = self.item_factors.row_slice(item_idx);

        let dot_product: f32 = user_row
            .iter()
            .zip(item_row.iter())
            .map(|(u, i)| u * i)
            .sum();

        self.global_mean + dot_product
    }

    /// Predict rating for user-item pair
    #[must_use]
    pub fn predict(&self, user_idx: usize, item_idx: usize) -> f32 {
        self.predict_internal(user_idx, item_idx).clamp(0.0, 5.0)
    }

    /// Get user latent factors
    #[must_use]
    pub fn get_user_factors(&self, user_idx: usize) -> Option<DenseVector> {
        if user_idx < self.user_factors.nrows() {
            Some(DenseVector::from_vec(self.user_factors.row_vec(user_idx)))
        } else {
            None
        }
    }

    /// Get item latent factors
    #[must_use]
    pub fn get_item_factors(&self, item_idx: usize) -> Option<DenseVector> {
        if item_idx < self.item_factors.nrows() {
            Some(DenseVector::from_vec(self.item_factors.row_vec(item_idx)))
        } else {
            None
        }
    }

    /// Initialize factors with small values
    fn initialize_factors(&mut self) {
        for i in 0..self.user_factors.nrows() {
            for j in 0..self.num_factors {
                self.user_factors.set(i, j, 0.1);
            }
        }

        for i in 0..self.item_factors.nrows() {
            for j in 0..self.num_factors {
                self.item_factors.set(i, j, 0.1);
            }
        }
    }

    /// Get number of factors
    #[must_use]
    pub fn num_factors(&self) -> usize {
        self.num_factors
    }

    /// Get global mean
    #[must_use]
    pub fn global_mean(&self) -> f32 {
        self.global_mean
    }

    /// Get number of users in the model
    #[must_use]
    pub fn num_users(&self) -> usize {
        self.user_factors.nrows()
    }

    /// Get number of items in the model
    #[must_use]
    pub fn num_items(&self) -> usize {
        self.item_factors.nrows()
    }
}

/// SVD trainer with hyperparameters
pub struct SvdTrainer {
    /// Number of latent factors
    num_factors: usize,
    /// Number of training epochs
    epochs: usize,
    /// Learning rate
    learning_rate: f32,
    /// Regularization parameter
    regularization: f32,
}

impl SvdTrainer {
    /// Create a new SVD trainer with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_factors: 20,
            epochs: 20,
            learning_rate: 0.005,
            regularization: 0.02,
        }
    }

    /// Set number of factors
    #[must_use]
    pub fn with_factors(mut self, num_factors: usize) -> Self {
        self.num_factors = num_factors;
        self
    }

    /// Set number of epochs
    #[must_use]
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set learning rate
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set regularization parameter
    #[must_use]
    pub fn with_regularization(mut self, regularization: f32) -> Self {
        self.regularization = regularization;
        self
    }

    /// Train an SVD model
    ///
    /// # Errors
    ///
    /// Returns an error if training fails
    pub fn train(
        &self,
        num_users: usize,
        num_items: usize,
        ratings: &[(usize, usize, f32)],
    ) -> RecommendResult<SvdModel> {
        let mut model = SvdModel::new(num_users, num_items, self.num_factors);
        model.train(
            ratings,
            self.epochs,
            self.learning_rate,
            self.regularization,
        )?;
        Ok(model)
    }
}

impl Default for SvdTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_model_creation() {
        let model = SvdModel::new(100, 200, 20);
        assert_eq!(model.num_factors(), 20);
        assert_eq!(model.num_users(), 100);
        assert_eq!(model.num_items(), 200);
    }

    #[test]
    fn test_svd_trainer_creation() {
        let trainer = SvdTrainer::new();
        assert_eq!(trainer.num_factors, 20);
        assert_eq!(trainer.epochs, 20);
    }

    #[test]
    fn test_svd_trainer_builder() {
        let trainer = SvdTrainer::new()
            .with_factors(10)
            .with_epochs(30)
            .with_learning_rate(0.01)
            .with_regularization(0.01);

        assert_eq!(trainer.num_factors, 10);
        assert_eq!(trainer.epochs, 30);
        assert!((trainer.learning_rate - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_svd_train() {
        let ratings = vec![(0, 0, 5.0), (0, 1, 3.0), (1, 0, 4.0), (1, 1, 2.0)];

        let trainer = SvdTrainer::new().with_epochs(10);
        let result = trainer.train(2, 2, &ratings);
        assert!(result.is_ok());

        if let Ok(model) = result {
            assert!(model.global_mean() > 0.0);
        }
    }

    #[test]
    fn test_svd_predict() {
        let mut model = SvdModel::new(2, 2, 5);
        model.global_mean = 3.5;

        let prediction = model.predict(0, 0);
        assert!((0.0..=5.0).contains(&prediction));
    }

    #[test]
    fn test_svd_get_factors() {
        let model = SvdModel::new(10, 10, 5);
        let user_factors = model.get_user_factors(0);
        assert!(user_factors.is_some());
        if let Some(factors) = user_factors {
            assert_eq!(factors.len(), 5);
        }

        let item_factors = model.get_item_factors(0);
        assert!(item_factors.is_some());
        if let Some(factors) = item_factors {
            assert_eq!(factors.len(), 5);
        }
    }
}
