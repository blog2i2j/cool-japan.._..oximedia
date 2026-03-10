//! Color-based search implementation.

use crate::error::SearchResult;
use crate::index::builder::Color;
use std::path::Path;
use uuid::Uuid;

/// Color search index
pub struct ColorIndex {
    index_path: std::path::PathBuf,
    colors: Vec<(Uuid, Vec<Color>)>,
}

impl ColorIndex {
    /// Create a new color index
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails
    pub fn new(index_path: &Path) -> SearchResult<Self> {
        if !index_path.exists() {
            std::fs::create_dir_all(index_path)?;
        }

        Ok(Self {
            index_path: index_path.to_path_buf(),
            colors: Vec::new(),
        })
    }

    /// Add colors for an asset
    ///
    /// # Errors
    ///
    /// Returns an error if indexing fails
    pub fn add_colors(&mut self, asset_id: Uuid, colors: &[Color]) -> SearchResult<()> {
        self.colors.push((asset_id, colors.to_vec()));
        Ok(())
    }

    /// Commit changes
    ///
    /// # Errors
    ///
    /// Returns an error if commit fails
    pub fn commit(&self) -> SearchResult<()> {
        Ok(())
    }

    /// Delete colors for an asset
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub fn delete(&mut self, asset_id: Uuid) -> SearchResult<()> {
        self.colors.retain(|(id, _)| *id != asset_id);
        Ok(())
    }
}

/// Color search engine.
///
/// Stores per-asset dominant colours and finds assets whose colours
/// fall within a configurable tolerance of a query colour using
/// Euclidean distance in RGB space.
pub struct ColorSearch {
    /// Per-asset colour database: `(asset_id, dominant_colours)`.
    entries: Vec<(Uuid, Vec<Color>)>,
}

impl ColorSearch {
    /// Create a new, empty colour search engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add dominant colours for an asset.
    pub fn add_asset(&mut self, asset_id: Uuid, colors: &[Color]) {
        self.entries.push((asset_id, colors.to_vec()));
    }

    /// Remove an asset from the search index.
    pub fn remove_asset(&mut self, asset_id: Uuid) {
        self.entries.retain(|(id, _)| *id != asset_id);
    }

    /// Search for assets that contain a colour within `tolerance`
    /// Euclidean distance of the query `(r, g, b)`.
    ///
    /// Returns asset IDs sorted by closest colour distance (ascending).
    ///
    /// # Errors
    ///
    /// Returns an error if search fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn search_by_color(&self, r: u8, g: u8, b: u8, tolerance: u8) -> SearchResult<Vec<Uuid>> {
        let tol_sq = f32::from(tolerance) * f32::from(tolerance);
        let qr = f32::from(r);
        let qg = f32::from(g);
        let qb = f32::from(b);

        let mut matches: Vec<(Uuid, f32)> = self
            .entries
            .iter()
            .filter_map(|(id, colors)| {
                // Find the minimum distance among this asset's dominant colours.
                let min_dist = colors
                    .iter()
                    .map(|c| {
                        let dr = qr - f32::from(c.r);
                        let dg = qg - f32::from(c.g);
                        let db = qb - f32::from(c.b);
                        dr * dr + dg * dg + db * db
                    })
                    .fold(f32::MAX, f32::min);

                if min_dist <= tol_sq {
                    Some((*id, min_dist))
                } else {
                    None
                }
            })
            .collect();

        // Sort by distance ascending (closest first).
        matches.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(matches.into_iter().map(|(id, _)| id).collect())
    }

    /// Return the number of indexed assets.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for ColorSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_search_empty() {
        let search = ColorSearch::new();
        let results = search
            .search_by_color(255, 0, 0, 10)
            .expect("should succeed in test");
        assert!(results.is_empty());
    }

    #[test]
    fn test_color_search_exact_match() {
        let mut search = ColorSearch::new();
        let id = Uuid::new_v4();
        search.add_asset(
            id,
            &[Color {
                r: 255,
                g: 0,
                b: 0,
                percentage: 100.0,
            }],
        );

        let results = search
            .search_by_color(255, 0, 0, 0)
            .expect("should succeed in test");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], id);
    }

    #[test]
    fn test_color_search_within_tolerance() {
        let mut search = ColorSearch::new();
        let id = Uuid::new_v4();
        // Colour (250, 5, 5) is close to (255, 0, 0).
        search.add_asset(
            id,
            &[Color {
                r: 250,
                g: 5,
                b: 5,
                percentage: 100.0,
            }],
        );

        // Euclidean distance = sqrt(25 + 25 + 25) ~= 8.66
        let results = search
            .search_by_color(255, 0, 0, 9)
            .expect("should succeed in test");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_color_search_outside_tolerance() {
        let mut search = ColorSearch::new();
        let id = Uuid::new_v4();
        search.add_asset(
            id,
            &[Color {
                r: 0,
                g: 255,
                b: 0,
                percentage: 100.0,
            }],
        );

        let results = search
            .search_by_color(255, 0, 0, 10)
            .expect("should succeed in test");
        assert!(results.is_empty());
    }

    #[test]
    fn test_color_search_sorted_by_distance() {
        let mut search = ColorSearch::new();
        let id_close = Uuid::new_v4();
        let id_far = Uuid::new_v4();
        search.add_asset(
            id_close,
            &[Color {
                r: 253,
                g: 0,
                b: 0,
                percentage: 100.0,
            }],
        );
        search.add_asset(
            id_far,
            &[Color {
                r: 240,
                g: 0,
                b: 0,
                percentage: 100.0,
            }],
        );

        let results = search
            .search_by_color(255, 0, 0, 20)
            .expect("should succeed in test");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], id_close);
        assert_eq!(results[1], id_far);
    }

    #[test]
    fn test_remove_asset() {
        let mut search = ColorSearch::new();
        let id = Uuid::new_v4();
        search.add_asset(
            id,
            &[Color {
                r: 100,
                g: 100,
                b: 100,
                percentage: 100.0,
            }],
        );
        assert_eq!(search.asset_count(), 1);
        search.remove_asset(id);
        assert_eq!(search.asset_count(), 0);
    }
}
