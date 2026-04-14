//! # oximedia-analytics
//!
//! Media engagement analytics for the OxiMedia Sovereign Media Framework.
//!
//! Provides viewer session tracking, audience retention curves, A/B testing,
//! multi-armed bandits, cohort analysis, funnel analysis, real-time aggregation,
//! time-series decomposition, EMA trend analysis, geo/device breakdowns,
//! reservoir-sampled heatmaps, CTR tracking, and engagement scoring — pure Rust.

pub mod ab_testing;
pub mod anomaly;
pub mod attribution;
pub mod bandit;
pub mod cohort;
pub mod ctr;
pub mod engagement;
pub mod error;
pub mod event_buffer;
pub mod fingerprint;
pub mod funnel;
pub mod geo_device;
pub mod heatmap;
pub mod multivariate;
pub mod percentile;
pub mod quantile;
pub mod realtime;
pub mod recommendation;
pub mod retention;
pub mod segment_retention;
pub mod session;
pub mod weighted_retention;

// ── Re-exports of key public types ─────────────────────────────────────────

pub use ab_testing::{
    assign_variant, bayesian_winner, winning_variant, winning_variant_with_alpha, AssignmentMethod,
    BayesianAbResult, Experiment, ExperimentResults, OptimisationMetric, Variant, VariantMetrics,
};
pub use bandit::{BanditArm, BanditStrategy, MultiArmedBandit, RegretTracker};
pub use cohort::{
    build_cohort_matrix, Cohort, CohortAnalyzer, CohortDefinition, CohortMatrix,
    CohortRetentionCell, CohortWindow, UserEvent, ViewerEvent,
};
pub use engagement::{
    compute_engagement, decompose_time_series, exponential_moving_average, linear_regression_slope,
    ContentEngagementScore, ContentRanker, DecomposedSeries, EmaConfig, EmaResult,
    EngagementComponents, EngagementTrend, EngagementWeights, SeasonalPeriod, TrendDirection,
};
pub use error::AnalyticsError;
pub use funnel::{
    compute_funnel, compute_loyalty, predict_churn, ChurnAssessment, ChurnConfig, ChurnRisk,
    FunnelAnalyzer, FunnelDefinition, FunnelMilestone, FunnelReport, FunnelResult, FunnelStep,
    FunnelStepDef, LoyaltyComponents, LoyaltyScore, LoyaltyWeights, SessionEvent,
};
pub use geo_device::{
    BreakdownAnalyzer, DeviceType, GeoDeviceReport, PeriodDelta, Region, SessionRecord,
    SliceComparison, SliceMetrics, TimestampedRecord,
};
pub use quantile::{percentiles, TDigest};
pub use realtime::{BucketMetrics, RealtimeEvent, SlidingWindowAggregator};
pub use retention::{
    average_view_duration, compare_to_benchmark, compute_retention, compute_retention_incremental,
    compute_segment_retention, drop_off_points, re_watch_segments, ContentSegment,
    IncrementalRetentionState, RetentionBenchmark, RetentionBucket, RetentionCurve,
    SegmentRetentionResult,
};
pub use session::{
    analyze_session, analyze_sessions_batch, attention_heatmap, build_playback_map,
    reservoir_sampled_heatmap, HeatPoint, PlaybackEvent, PlaybackMap, ReservoirHeatmapConfig,
    SampledHeatmap, SessionMetrics, ViewerSession,
};
