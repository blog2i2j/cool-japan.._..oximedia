//! Migration planning for format obsolescence

use super::MigrationPriority;
use crate::{PreservationFormat, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Migration strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationStrategy {
    /// Direct conversion
    Direct,
    /// Two-step migration via intermediate format
    TwoStep {
        /// Intermediate format
        intermediate: String,
    },
    /// Keep original and create new copy
    KeepBoth,
    /// Replace original with new format
    Replace,
}

/// Migration plan for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Source file
    pub source: PathBuf,
    /// Source format
    pub source_format: String,
    /// Target format
    pub target_format: PreservationFormat,
    /// Migration strategy
    pub strategy: MigrationStrategy,
    /// Priority
    pub priority: MigrationPriority,
    /// Estimated risk of waiting
    pub risk_score: f32,
    /// Recommended completion date
    pub recommended_date: Option<chrono::DateTime<chrono::Utc>>,
}

/// Migration planner
pub struct MigrationPlanner {
    risk_threshold: f32,
}

impl Default for MigrationPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl MigrationPlanner {
    /// Create a new migration planner
    #[must_use]
    pub fn new() -> Self {
        Self {
            risk_threshold: 0.5,
        }
    }

    /// Set the risk threshold for high-priority migrations
    #[must_use]
    pub fn with_risk_threshold(mut self, threshold: f32) -> Self {
        self.risk_threshold = threshold;
        self
    }

    /// Plan migration for a file
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails
    pub fn plan_migration(
        &self,
        source: PathBuf,
        source_format: &str,
        target_format: PreservationFormat,
    ) -> Result<MigrationPlan> {
        let risk_score = self.assess_format_risk(source_format);
        let priority = self.determine_priority(risk_score);
        let strategy = self.determine_strategy(source_format, &target_format);

        let recommended_date = if priority == MigrationPriority::Critical {
            Some(chrono::Utc::now())
        } else {
            Some(chrono::Utc::now() + chrono::Duration::days(30))
        };

        Ok(MigrationPlan {
            source,
            source_format: source_format.to_string(),
            target_format,
            strategy,
            priority,
            risk_score,
            recommended_date,
        })
    }

    /// Assess risk score for a format (0.0 = low risk, 1.0 = high risk)
    fn assess_format_risk(&self, format: &str) -> f32 {
        // Simplified risk assessment based on format
        match format.to_lowercase().as_str() {
            // High risk: proprietary or obsolete formats
            "wmv" | "asf" | "flv" | "rm" | "3gp" => 0.9,
            "avi" | "mov" => 0.7, // Containers with potential patent issues
            "mp4" | "m4v" => 0.5, // Widely supported but patent encumbered
            // Medium risk
            "webm" => 0.3, // Open but relatively new
            // Low risk: preservation formats
            "mkv" | "flac" | "wav" => 0.1,
            _ => 0.5, // Unknown formats get medium risk
        }
    }

    fn determine_priority(&self, risk_score: f32) -> MigrationPriority {
        if risk_score >= 0.8 {
            MigrationPriority::Critical
        } else if risk_score >= 0.6 {
            MigrationPriority::High
        } else if risk_score >= 0.4 {
            MigrationPriority::Medium
        } else {
            MigrationPriority::Low
        }
    }

    fn determine_strategy(
        &self,
        _source_format: &str,
        _target: &PreservationFormat,
    ) -> MigrationStrategy {
        // For now, use direct migration and keep both files
        MigrationStrategy::KeepBoth
    }

    /// Plan migrations for multiple files
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails
    pub fn plan_batch(
        &self,
        files: Vec<(PathBuf, String, PreservationFormat)>,
    ) -> Result<Vec<MigrationPlan>> {
        files
            .into_iter()
            .map(|(path, source_fmt, target_fmt)| {
                self.plan_migration(path, &source_fmt, target_fmt)
            })
            .collect()
    }

    /// Sort plans by priority
    #[must_use]
    pub fn prioritize_plans(&self, mut plans: Vec<MigrationPlan>) -> Vec<MigrationPlan> {
        plans.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then(
                b.risk_score
                    .partial_cmp(&a.risk_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        plans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_migration() {
        let planner = MigrationPlanner::new();
        let plan = planner
            .plan_migration(
                PathBuf::from("test.wmv"),
                "wmv",
                PreservationFormat::VideoFfv1Mkv,
            )
            .expect("operation should succeed");

        assert_eq!(plan.source_format, "wmv");
        assert_eq!(plan.target_format, PreservationFormat::VideoFfv1Mkv);
        assert!(plan.risk_score > 0.8);
        assert_eq!(plan.priority, MigrationPriority::Critical);
    }

    #[test]
    fn test_format_risk_assessment() {
        let planner = MigrationPlanner::new();

        assert!(planner.assess_format_risk("wmv") > 0.8);
        assert!(planner.assess_format_risk("mkv") < 0.2);
    }

    #[test]
    fn test_prioritize_plans() {
        let planner = MigrationPlanner::new();
        let plans = vec![
            MigrationPlan {
                source: PathBuf::from("low.mkv"),
                source_format: "mkv".to_string(),
                target_format: PreservationFormat::VideoFfv1Mkv,
                strategy: MigrationStrategy::Direct,
                priority: MigrationPriority::Low,
                risk_score: 0.1,
                recommended_date: None,
            },
            MigrationPlan {
                source: PathBuf::from("critical.wmv"),
                source_format: "wmv".to_string(),
                target_format: PreservationFormat::VideoFfv1Mkv,
                strategy: MigrationStrategy::Direct,
                priority: MigrationPriority::Critical,
                risk_score: 0.9,
                recommended_date: None,
            },
        ];

        let sorted = planner.prioritize_plans(plans);
        assert_eq!(sorted[0].priority, MigrationPriority::Critical);
        assert_eq!(sorted[1].priority, MigrationPriority::Low);
    }
}
