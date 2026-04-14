//! Multivariate testing: test multiple variables (factors) simultaneously.
//!
//! Standard A/B testing compares a single treatment variable.  Multivariate
//! (MVT) testing varies several independent factors at once and estimates the
//! *interaction effect* between them, enabling richer experimentation with
//! fewer total sessions than running every pair in separate A/B tests.
//!
//! ## Design
//! This module implements a **full-factorial** MVT design: every combination of
//! factor levels is a distinct treatment *cell*.  Cells are assigned
//! deterministically via FNV-1a hashing so that the same viewer always lands in
//! the same cell.
//!
//! ## Statistical analysis
//! - **Conversion rates** are estimated per cell with Wilson-score confidence
//!   intervals (same approach as the `ctr` module).
//! - **Main effects** are computed as the average conversion-rate difference
//!   attributable to each factor level, marginalised over all other factors.
//! - **Winning cell** is selected by the highest conversion rate once a minimum
//!   sample-size threshold is satisfied.
//!
//! ## Example
//! ```rust
//! use oximedia_analytics::multivariate::{
//!     Factor, FactorLevel, MultivariateExperiment, MultivariateMetrics,
//! };
//!
//! let thumbnail = Factor {
//!     name: "thumbnail".to_string(),
//!     levels: vec![
//!         FactorLevel { id: "bright".to_string(), label: "Bright".to_string() },
//!         FactorLevel { id: "dark".to_string(),   label: "Dark".to_string() },
//!     ],
//! };
//! let title = Factor {
//!     name: "title".to_string(),
//!     levels: vec![
//!         FactorLevel { id: "short".to_string(), label: "Short".to_string() },
//!         FactorLevel { id: "long".to_string(),  label: "Long".to_string() },
//!     ],
//! };
//! let experiment = MultivariateExperiment::new(
//!     "mvt_001".to_string(),
//!     "Homepage Thumbnail × Title".to_string(),
//!     vec![thumbnail, title],
//!     50,    // min samples per cell
//! ).expect("experiment");
//!
//! // Assign a viewer.
//! let cell = experiment.assign_cell("user_42").expect("assignment");
//! assert_eq!(cell.assignments.len(), 2); // one level per factor
//! ```

use std::collections::HashMap;

use crate::error::AnalyticsError;

// ─── Factor definitions ───────────────────────────────────────────────────────

/// One level (value) for a single experimental factor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorLevel {
    /// Short machine-readable identifier (e.g. `"bright"`, `"short"`).
    pub id: String,
    /// Human-readable label for reporting (e.g. `"Bright thumbnail"`).
    pub label: String,
}

/// A single experimental factor with its possible levels.
#[derive(Debug, Clone)]
pub struct Factor {
    /// Machine-readable factor name (e.g. `"thumbnail"`, `"title_copy"`).
    pub name: String,
    /// Ordered list of levels.  Must contain at least two entries.
    pub levels: Vec<FactorLevel>,
}

impl Factor {
    /// Number of levels in this factor.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

// ─── Cell ─────────────────────────────────────────────────────────────────────

/// A single treatment cell: a complete assignment of levels to all factors.
///
/// `assignments[i]` is the level index (into `factor[i].levels`) assigned
/// in this cell.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cell {
    /// Per-factor level indices.
    pub assignments: Vec<usize>,
}

impl Cell {
    /// Return a human-readable label `"f0_level × f1_level × …"`.
    pub fn label(&self, factors: &[Factor]) -> String {
        self.assignments
            .iter()
            .enumerate()
            .map(|(fi, &li)| {
                factors
                    .get(fi)
                    .and_then(|f| f.levels.get(li))
                    .map(|l| l.id.as_str())
                    .unwrap_or("?")
            })
            .collect::<Vec<_>>()
            .join(" × ")
    }
}

// ─── Experiment ───────────────────────────────────────────────────────────────

/// A full-factorial multivariate experiment.
#[derive(Debug, Clone)]
pub struct MultivariateExperiment {
    /// Unique experiment identifier.
    pub id: String,
    /// Human-readable experiment name.
    pub name: String,
    /// Ordered list of independent factors.
    pub factors: Vec<Factor>,
    /// Minimum number of impressions per cell before declaring a winner.
    pub min_samples_per_cell: u32,
    /// Total number of cells = product of all factor level counts.
    pub cell_count: usize,
}

impl MultivariateExperiment {
    /// Create a new full-factorial multivariate experiment.
    ///
    /// # Errors
    /// - [`AnalyticsError::InvalidInput`] — fewer than two factors, any factor
    ///   with fewer than two levels, or `min_samples_per_cell` is zero.
    pub fn new(
        id: String,
        name: String,
        factors: Vec<Factor>,
        min_samples_per_cell: u32,
    ) -> Result<Self, AnalyticsError> {
        if factors.len() < 2 {
            return Err(AnalyticsError::InvalidInput(
                "multivariate experiment requires at least 2 factors".to_string(),
            ));
        }
        for f in &factors {
            if f.levels.len() < 2 {
                return Err(AnalyticsError::InvalidInput(format!(
                    "factor '{}' must have at least 2 levels",
                    f.name
                )));
            }
        }
        if min_samples_per_cell == 0 {
            return Err(AnalyticsError::InvalidInput(
                "min_samples_per_cell must be > 0".to_string(),
            ));
        }

        let cell_count = factors.iter().map(|f| f.level_count()).product();

        Ok(Self {
            id,
            name,
            factors,
            min_samples_per_cell,
            cell_count,
        })
    }

    /// Enumerate all cells in the experiment (row-major order).
    pub fn all_cells(&self) -> Vec<Cell> {
        if self.factors.is_empty() {
            return Vec::new();
        }
        let mut cells: Vec<Cell> = vec![Cell {
            assignments: vec![],
        }];
        for factor in &self.factors {
            let mut expanded = Vec::with_capacity(cells.len() * factor.level_count());
            for cell in &cells {
                for li in 0..factor.level_count() {
                    let mut new_cell = cell.clone();
                    new_cell.assignments.push(li);
                    expanded.push(new_cell);
                }
            }
            cells = expanded;
        }
        cells
    }

    /// Deterministically assign a viewer to a cell using FNV-1a hashing.
    ///
    /// The same `user_id` always maps to the same cell within this experiment.
    ///
    /// Returns the `Cell` assignment.
    pub fn assign_cell(&self, user_id: &str) -> Result<Cell, AnalyticsError> {
        if self.cell_count == 0 {
            return Err(AnalyticsError::InvalidInput(
                "experiment has no cells".to_string(),
            ));
        }

        let hash = fnv1a_32(user_id.as_bytes());
        let cell_idx = (hash as usize) % self.cell_count;

        // Decompose cell_idx into per-factor level indices (little-endian).
        let mut assignments = Vec::with_capacity(self.factors.len());
        let mut remaining = cell_idx;
        for factor in &self.factors {
            let li = remaining % factor.level_count();
            assignments.push(li);
            remaining /= factor.level_count();
        }

        Ok(Cell { assignments })
    }
}

// ─── Metrics collection ───────────────────────────────────────────────────────

/// Observed metrics for a single cell.
#[derive(Debug, Clone, Default)]
pub struct CellMetrics {
    /// Total number of impressions (exposures) in this cell.
    pub impressions: u64,
    /// Total number of conversions (clicks, completions, etc.) in this cell.
    pub conversions: u64,
}

impl CellMetrics {
    /// Conversion rate (conversions / impressions).  Returns `0.0` when
    /// `impressions` is zero.
    pub fn conversion_rate(&self) -> f64 {
        if self.impressions == 0 {
            0.0
        } else {
            self.conversions as f64 / self.impressions as f64
        }
    }

    /// Wilson-score confidence interval for the conversion rate at the given
    /// significance level.
    ///
    /// `z` is the normal quantile for the desired confidence (e.g. `1.96` for
    /// 95 %).  Returns `(lower, upper)` in `[0.0, 1.0]`.
    pub fn wilson_interval(&self, z: f64) -> (f64, f64) {
        if self.impressions == 0 {
            return (0.0, 1.0);
        }
        let n = self.impressions as f64;
        let p = self.conversions as f64 / n;
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let centre = (p + z2 / (2.0 * n)) / denom;
        let margin = (z / denom) * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt();
        ((centre - margin).max(0.0), (centre + margin).min(1.0))
    }
}

/// Aggregated metrics across all cells in a multivariate experiment.
pub struct MultivariateMetrics {
    /// Per-cell raw metrics; keyed by cell (level-index vector).
    pub cell_metrics: HashMap<Cell, CellMetrics>,
    experiment: MultivariateExperiment,
}

impl MultivariateMetrics {
    /// Create an empty metrics collector for `experiment`.
    pub fn new(experiment: MultivariateExperiment) -> Self {
        let cells = experiment.all_cells();
        let cell_metrics: HashMap<Cell, CellMetrics> = cells
            .into_iter()
            .map(|c| (c, CellMetrics::default()))
            .collect();
        Self {
            cell_metrics,
            experiment,
        }
    }

    /// Record one impression (exposure) for `user_id`.
    ///
    /// The user is assigned to their cell deterministically and their
    /// impression counter is incremented.
    ///
    /// # Errors
    /// Propagates errors from [`MultivariateExperiment::assign_cell`].
    pub fn record_impression(&mut self, user_id: &str) -> Result<(), AnalyticsError> {
        let cell = self.experiment.assign_cell(user_id)?;
        self.cell_metrics.entry(cell).or_default().impressions += 1;
        Ok(())
    }

    /// Record a conversion for `user_id`.
    ///
    /// # Errors
    /// Propagates errors from [`MultivariateExperiment::assign_cell`].
    pub fn record_conversion(&mut self, user_id: &str) -> Result<(), AnalyticsError> {
        let cell = self.experiment.assign_cell(user_id)?;
        self.cell_metrics.entry(cell).or_default().conversions += 1;
        Ok(())
    }

    /// Return the cell with the highest conversion rate, provided it has
    /// accumulated at least `min_samples_per_cell` impressions.
    ///
    /// Returns `None` if no cell has sufficient data.
    pub fn winning_cell(&self) -> Option<(&Cell, &CellMetrics)> {
        let min = self.experiment.min_samples_per_cell as u64;
        self.cell_metrics
            .iter()
            .filter(|(_, m)| m.impressions >= min)
            .max_by(|(_, a), (_, b)| {
                a.conversion_rate()
                    .partial_cmp(&b.conversion_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Compute the *main effect* for factor `factor_idx` at level `level_idx`.
    ///
    /// The main effect is the average conversion-rate of all cells that have
    /// `factor[factor_idx] = level_idx`, minus the grand mean conversion rate.
    ///
    /// Returns `None` when no cells have impressions or the factor/level index
    /// is out of bounds.
    pub fn main_effect(&self, factor_idx: usize, level_idx: usize) -> Option<f64> {
        let factor = self.experiment.factors.get(factor_idx)?;
        if level_idx >= factor.level_count() {
            return None;
        }

        // Cells that contain factor_idx → level_idx.
        let matching: Vec<f64> = self
            .cell_metrics
            .iter()
            .filter(|(cell, _)| cell.assignments.get(factor_idx) == Some(&level_idx))
            .map(|(_, m)| m.conversion_rate())
            .collect();

        if matching.is_empty() {
            return None;
        }

        let mean_for_level: f64 = matching.iter().sum::<f64>() / matching.len() as f64;

        // Grand mean.
        let all_rates: Vec<f64> = self
            .cell_metrics
            .values()
            .map(|m| m.conversion_rate())
            .collect();
        let grand_mean: f64 = all_rates.iter().sum::<f64>() / all_rates.len() as f64;

        Some(mean_for_level - grand_mean)
    }

    /// Summarise all main effects per factor and level.
    ///
    /// Returns `Vec<(factor_name, level_id, main_effect)>`.
    pub fn all_main_effects(&self) -> Vec<(String, String, f64)> {
        let mut results = Vec::new();
        for (fi, factor) in self.experiment.factors.iter().enumerate() {
            for (li, level) in factor.levels.iter().enumerate() {
                if let Some(effect) = self.main_effect(fi, li) {
                    results.push((factor.name.clone(), level.id.clone(), effect));
                }
            }
        }
        results
    }

    /// Borrow the underlying experiment definition.
    pub fn experiment(&self) -> &MultivariateExperiment {
        &self.experiment
    }
}

// ─── FNV-1a helper ────────────────────────────────────────────────────────────

fn fnv1a_32(data: &[u8]) -> u32 {
    const FNV_OFFSET: u32 = 2_166_136_261;
    const FNV_PRIME: u32 = 16_777_619;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_factor_experiment() -> MultivariateExperiment {
        MultivariateExperiment::new(
            "exp1".to_string(),
            "Thumbnail × Title".to_string(),
            vec![
                Factor {
                    name: "thumbnail".to_string(),
                    levels: vec![
                        FactorLevel {
                            id: "bright".to_string(),
                            label: "Bright".to_string(),
                        },
                        FactorLevel {
                            id: "dark".to_string(),
                            label: "Dark".to_string(),
                        },
                    ],
                },
                Factor {
                    name: "title".to_string(),
                    levels: vec![
                        FactorLevel {
                            id: "short".to_string(),
                            label: "Short".to_string(),
                        },
                        FactorLevel {
                            id: "long".to_string(),
                            label: "Long".to_string(),
                        },
                    ],
                },
            ],
            10,
        )
        .expect("valid experiment")
    }

    #[test]
    fn cell_count_is_product_of_levels() {
        let exp = two_factor_experiment();
        assert_eq!(exp.cell_count, 4, "2 × 2 = 4 cells");
    }

    #[test]
    fn all_cells_enumerates_all_combinations() {
        let exp = two_factor_experiment();
        let cells = exp.all_cells();
        assert_eq!(cells.len(), 4);
        // Each combination should be unique.
        let unique: std::collections::HashSet<_> = cells.iter().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn assign_cell_is_deterministic() {
        let exp = two_factor_experiment();
        let cell_a = exp.assign_cell("user_123").expect("assignment");
        let cell_b = exp.assign_cell("user_123").expect("assignment");
        assert_eq!(cell_a, cell_b, "same user must always get same cell");
    }

    #[test]
    fn assign_cell_covers_all_cells_with_enough_users() {
        let exp = two_factor_experiment();
        let mut seen = std::collections::HashSet::new();
        for i in 0..1000u32 {
            let user = format!("u{i}");
            let cell = exp.assign_cell(&user).expect("assignment");
            seen.insert(cell.assignments.clone());
        }
        assert_eq!(
            seen.len(),
            exp.cell_count,
            "all cells should be assigned at least once across 1000 users"
        );
    }

    #[test]
    fn record_impression_and_conversion() {
        let exp = two_factor_experiment();
        let mut metrics = MultivariateMetrics::new(exp);

        metrics.record_impression("alice").expect("impression");
        metrics.record_conversion("alice").expect("conversion");

        let cell = metrics.experiment.assign_cell("alice").expect("cell");
        let m = &metrics.cell_metrics[&cell];
        assert_eq!(m.impressions, 1);
        assert_eq!(m.conversions, 1);
        assert!((m.conversion_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn winning_cell_selected_by_conversion_rate() {
        let exp = two_factor_experiment();
        let mut metrics = MultivariateMetrics::new(exp.clone());

        // Populate all cells with 20 impressions.
        let cells = exp.all_cells();
        // Fake fill: manipulate via record_impression by finding users that map to each cell.
        // Use many random-ish user IDs and collect 20 per cell.
        let mut cell_user_map: HashMap<Vec<usize>, Vec<String>> = HashMap::new();
        for i in 0..10_000u32 {
            let user = format!("fill_user_{i}");
            let cell = exp.assign_cell(&user).expect("cell");
            let v = cell_user_map.entry(cell.assignments.clone()).or_default();
            if v.len() < 25 {
                v.push(user);
            }
        }

        for (_assignments, users) in &cell_user_map {
            for user in &users[..users.len().min(20)] {
                metrics.record_impression(user).expect("imp");
            }
        }

        // Pick the first cell; add conversions to it.
        let target_cell = &cells[0];
        let target_users = &cell_user_map[&target_cell.assignments];
        for user in target_users.iter().take(18) {
            metrics.record_conversion(user).expect("conv");
        }

        let winner = metrics.winning_cell();
        assert!(winner.is_some(), "should find a winning cell");
    }

    #[test]
    fn main_effect_sign_makes_sense() {
        // Assign users to cells and simulate that "bright" thumbnail has higher CTR.
        let exp = two_factor_experiment();
        let mut metrics = MultivariateMetrics::new(exp.clone());

        let mut bright_users: Vec<String> = Vec::new();
        let mut dark_users: Vec<String> = Vec::new();

        for i in 0..5_000u32 {
            let user = format!("me_{i}");
            let cell = exp.assign_cell(&user).expect("cell");
            // thumbnail is factor 0; bright = level 0
            if cell.assignments[0] == 0 {
                bright_users.push(user);
            } else {
                dark_users.push(user);
            }
        }

        // Give bright 200 impressions, 160 conversions (80%).
        for u in bright_users.iter().take(200) {
            metrics.record_impression(u).expect("imp");
        }
        for u in bright_users.iter().take(160) {
            metrics.record_conversion(u).expect("conv");
        }

        // Give dark 200 impressions, 60 conversions (30%).
        for u in dark_users.iter().take(200) {
            metrics.record_impression(u).expect("imp");
        }
        for u in dark_users.iter().take(60) {
            metrics.record_conversion(u).expect("conv");
        }

        let bright_effect = metrics.main_effect(0, 0);
        let dark_effect = metrics.main_effect(0, 1);

        assert!(bright_effect.is_some() && dark_effect.is_some());
        assert!(
            bright_effect.unwrap() > dark_effect.unwrap(),
            "bright effect ({:?}) should exceed dark effect ({:?})",
            bright_effect,
            dark_effect,
        );
    }

    #[test]
    fn experiment_requires_at_least_two_factors() {
        let result = MultivariateExperiment::new(
            "e1".to_string(),
            "single factor".to_string(),
            vec![Factor {
                name: "only".to_string(),
                levels: vec![
                    FactorLevel {
                        id: "a".to_string(),
                        label: "A".to_string(),
                    },
                    FactorLevel {
                        id: "b".to_string(),
                        label: "B".to_string(),
                    },
                ],
            }],
            10,
        );
        assert!(result.is_err(), "single-factor experiment must be rejected");
    }

    #[test]
    fn cell_label_produces_readable_string() {
        let exp = two_factor_experiment();
        let cell = Cell {
            assignments: vec![0, 1],
        };
        let label = cell.label(&exp.factors);
        assert!(label.contains("bright"), "label should mention 'bright'");
        assert!(label.contains("long"), "label should mention 'long'");
    }
}
