//! User-item matrix for collaborative filtering.

use crate::dense_linalg::DenseMatrix;
use crate::error::RecommendResult;
use crate::{ContentMetadata, Recommendation, RecommendationReason, RecommendationRequest};
use std::collections::HashMap;
use uuid::Uuid;

/// User-item interaction matrix
#[derive(Debug, Clone)]
pub struct UserItemMatrix {
    /// Matrix data (users x items)
    data: DenseMatrix,
    /// User ID to index mapping
    user_to_index: HashMap<Uuid, usize>,
    /// Item ID to index mapping
    item_to_index: HashMap<Uuid, usize>,
    /// Index to user ID mapping
    index_to_user: Vec<Uuid>,
    /// Index to item ID mapping
    index_to_item: Vec<Uuid>,
}

impl UserItemMatrix {
    /// Create a new user-item matrix
    #[must_use]
    pub fn new(num_users: usize, num_items: usize) -> Self {
        Self {
            data: DenseMatrix::zeros(num_users, num_items),
            user_to_index: HashMap::new(),
            item_to_index: HashMap::new(),
            index_to_user: Vec::new(),
            index_to_item: Vec::new(),
        }
    }

    /// Add a user
    pub fn add_user(&mut self, user_id: Uuid) -> usize {
        if let Some(&index) = self.user_to_index.get(&user_id) {
            return index;
        }

        let index = self.index_to_user.len();
        self.user_to_index.insert(user_id, index);
        self.index_to_user.push(user_id);

        // Expand matrix if needed
        if index >= self.data.nrows() {
            let new_rows = index + 1 - self.data.nrows();
            let zeros = DenseMatrix::zeros(new_rows, self.data.ncols());
            self.data = self.data.concat_rows(&zeros);
        }

        index
    }

    /// Add an item
    pub fn add_item(&mut self, item_id: Uuid) -> usize {
        if let Some(&index) = self.item_to_index.get(&item_id) {
            return index;
        }

        let index = self.index_to_item.len();
        self.item_to_index.insert(item_id, index);
        self.index_to_item.push(item_id);

        // Expand matrix if needed
        if index >= self.data.ncols() {
            let new_cols = index + 1 - self.data.ncols();
            let zeros = DenseMatrix::zeros(self.data.nrows(), new_cols);
            self.data = self.data.concat_cols(&zeros);
        }

        index
    }

    /// Set rating for user-item pair
    pub fn set_rating(&mut self, user_id: Uuid, item_id: Uuid, rating: f32) {
        let user_idx = self.add_user(user_id);
        let item_idx = self.add_item(item_id);
        self.data.set(user_idx, item_idx, rating);
    }

    /// Get rating for user-item pair
    #[must_use]
    pub fn get_rating(&self, user_id: Uuid, item_id: Uuid) -> Option<f32> {
        let user_idx = self.user_to_index.get(&user_id)?;
        let item_idx = self.item_to_index.get(&item_id)?;
        Some(self.data.get(*user_idx, *item_idx))
    }

    /// Get user's ratings vector
    #[must_use]
    pub fn get_user_ratings(&self, user_id: Uuid) -> Option<Vec<f32>> {
        let user_idx = self.user_to_index.get(&user_id)?;
        Some(self.data.row_vec(*user_idx))
    }

    /// Get item's ratings vector
    #[must_use]
    pub fn get_item_ratings(&self, item_id: Uuid) -> Option<Vec<f32>> {
        let item_idx = self.item_to_index.get(&item_id)?;
        Some(self.data.col_vec(*item_idx))
    }

    /// Get number of rows in the underlying data matrix
    #[must_use]
    pub fn data_nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Get number of columns in the underlying data matrix
    #[must_use]
    pub fn data_ncols(&self) -> usize {
        self.data.ncols()
    }

    /// Get a value from the underlying data matrix by indices
    #[must_use]
    pub fn data_get(&self, row: usize, col: usize) -> f32 {
        self.data.get(row, col)
    }

    /// Get a row from the underlying data matrix as a Vec
    #[must_use]
    pub fn data_row_vec(&self, row: usize) -> Vec<f32> {
        self.data.row_vec(row)
    }

    /// Get number of users
    #[must_use]
    pub fn num_users(&self) -> usize {
        self.index_to_user.len()
    }

    /// Get number of items
    #[must_use]
    pub fn num_items(&self) -> usize {
        self.index_to_item.len()
    }

    /// Get item ID by index
    #[must_use]
    pub fn get_item_id(&self, index: usize) -> Option<Uuid> {
        self.index_to_item.get(index).copied()
    }

    /// Get user ID by index
    #[must_use]
    pub fn get_user_id(&self, index: usize) -> Option<Uuid> {
        self.index_to_user.get(index).copied()
    }

    /// Find items rated by user
    #[must_use]
    pub fn get_rated_items(&self, user_id: Uuid) -> Vec<(Uuid, f32)> {
        let Some(&user_idx) = self.user_to_index.get(&user_id) else {
            return Vec::new();
        };

        let row = self.data.row_vec(user_idx);
        row.iter()
            .enumerate()
            .filter(|(_, &rating)| rating > 0.0)
            .filter_map(|(item_idx, &rating)| {
                self.index_to_item
                    .get(item_idx)
                    .map(|&item_id| (item_id, rating))
            })
            .collect()
    }
}

/// Collaborative filtering engine
pub struct CollaborativeEngine {
    /// User-item matrix
    matrix: UserItemMatrix,
    /// Content metadata
    content_metadata: HashMap<Uuid, ContentMetadata>,
    /// K-nearest neighbors calculator
    knn: super::knn::KnnCalculator,
}

impl CollaborativeEngine {
    /// Create a new collaborative engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            matrix: UserItemMatrix::new(0, 0),
            content_metadata: HashMap::new(),
            knn: super::knn::KnnCalculator::new(10),
        }
    }

    /// Add a rating
    pub fn add_rating(&mut self, user_id: Uuid, content_id: Uuid, rating: f32) {
        self.matrix.set_rating(user_id, content_id, rating);
    }

    /// Add content metadata
    pub fn add_content(&mut self, content_id: Uuid, metadata: ContentMetadata) {
        self.content_metadata.insert(content_id, metadata);
    }

    /// Get collaborative recommendations
    ///
    /// # Errors
    ///
    /// Returns an error if recommendation generation fails
    pub fn recommend(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        // Find similar users
        let similar_users = self
            .knn
            .find_similar_users(&self.matrix, request.user_id, 20)?;

        // Get items rated by similar users
        let mut candidate_items: HashMap<Uuid, f32> = HashMap::new();

        for (similar_user, similarity) in similar_users {
            let rated_items = self.matrix.get_rated_items(similar_user);
            for (item_id, rating) in rated_items {
                // Skip items already rated by the user
                if self.matrix.get_rating(request.user_id, item_id).is_some() {
                    continue;
                }

                *candidate_items.entry(item_id).or_insert(0.0) += rating * similarity;
            }
        }

        // Convert to recommendations
        let mut recommendations: Vec<Recommendation> = candidate_items
            .into_iter()
            .filter_map(|(content_id, score)| {
                self.content_metadata
                    .get(&content_id)
                    .map(|metadata| Recommendation {
                        content_id,
                        score,
                        rank: 0,
                        reasons: vec![RecommendationReason::CollaborativeFiltering {
                            confidence: score,
                        }],
                        metadata: metadata.clone(),
                        explanation: None,
                    })
            })
            .collect();

        // Sort by score
        recommendations.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (idx, rec) in recommendations.iter_mut().enumerate() {
            rec.rank = idx + 1;
        }

        recommendations.truncate(request.limit);

        Ok(recommendations)
    }
}

impl Default for CollaborativeEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_item_matrix_creation() {
        let matrix = UserItemMatrix::new(10, 20);
        assert_eq!(matrix.data_nrows(), 10);
        assert_eq!(matrix.data_ncols(), 20);
    }

    #[test]
    fn test_add_user() {
        let mut matrix = UserItemMatrix::new(0, 0);
        let user_id = Uuid::new_v4();
        let index = matrix.add_user(user_id);
        assert_eq!(index, 0);

        let index2 = matrix.add_user(user_id);
        assert_eq!(index2, 0); // Same index for same user
    }

    #[test]
    fn test_add_item() {
        let mut matrix = UserItemMatrix::new(0, 0);
        let item_id = Uuid::new_v4();
        let index = matrix.add_item(item_id);
        assert_eq!(index, 0);
    }

    #[test]
    fn test_set_get_rating() {
        let mut matrix = UserItemMatrix::new(0, 0);
        let user_id = Uuid::new_v4();
        let item_id = Uuid::new_v4();

        matrix.set_rating(user_id, item_id, 4.5);
        let rating = matrix.get_rating(user_id, item_id);
        assert_eq!(rating, Some(4.5));
    }

    #[test]
    fn test_collaborative_engine_creation() {
        let engine = CollaborativeEngine::new();
        assert_eq!(engine.matrix.num_users(), 0);
        assert_eq!(engine.matrix.num_items(), 0);
    }

    #[test]
    fn test_add_rating_to_engine() {
        let mut engine = CollaborativeEngine::new();
        let user_id = Uuid::new_v4();
        let content_id = Uuid::new_v4();

        engine.add_rating(user_id, content_id, 5.0);
        let rating = engine.matrix.get_rating(user_id, content_id);
        assert_eq!(rating, Some(5.0));
    }
}
