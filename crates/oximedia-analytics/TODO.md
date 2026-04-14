# oximedia-analytics TODO

## Current Status
- 22 modules: `session`, `retention`, `ab_testing`, `engagement`, `error`, `bandit`, `cohort`,
  `funnel`, `quantile`, `attribution`, `realtime`, `geo_device`, `anomaly`, `ctr`,
  `event_buffer`, `fingerprint`, `heatmap`, `multivariate`, `percentile`, `recommendation`,
  `segment_retention`, `weighted_retention`
- 23 source files total
- 327 unit tests passing (+ 6 doc-tests)
- Viewer session tracking with playback events, attention heatmaps, reservoir-sampled heatmaps
- Audience retention curves, segment retention, weighted demographic retention, incremental computation
- A/B testing (frequentist + Bayesian), multi-armed bandit (epsilon-greedy + Thompson sampling)
- Engagement scoring, linear regression + EMA trend analysis, time-series decomposition
- Cohort analysis, funnel analysis, churn prediction, loyalty scoring
- Real-time sliding-window aggregation, CTR tracking, geographic/device breakdowns
- Watch-time attribution, multivariate testing, content recommendation

## Enhancements
- [ ] Add configurable significance level (alpha) to `winning_variant` in `ab_testing` (currently hardcoded)
- [ ] Implement Bayesian A/B testing as alternative to frequentist z-test in `ab_testing`
- [ ] Add multi-armed bandit (epsilon-greedy, Thompson sampling) for adaptive experiments in `ab_testing`
- [ ] Implement segment-level retention analysis in `retention` (retention by content chapter/segment)
- [ ] Add weighted retention curves in `retention` accounting for viewer demographics
- [ ] Implement time-series decomposition (trend + seasonality) in `engagement` trend analysis
- [x] Add exponential moving average option alongside linear regression in `linear_regression_slope`
- [ ] Implement funnel analysis in `session` (track viewer progression through content milestones)
- [ ] Add session replay reconstruction from `PlaybackEvent` sequences for debugging

## New Features
- [ ] Add cohort analysis module: group viewers by first-view date and track retention over time
- [ ] Implement churn prediction based on engagement score decline patterns
- [ ] Add real-time analytics aggregation: sliding window metrics (concurrent viewers, bitrate stats)
- [ ] Implement content recommendation scoring based on engagement similarity
- [x] Add geographic/device breakdowns for session metrics
- [ ] Implement watch time attribution: allocate credit to content segments for total engagement
- [ ] Add multivariate testing support (test multiple variables simultaneously) in `ab_testing`
- [ ] Implement click-through rate (CTR) tracking for thumbnails and previews
- [ ] Add viewer loyalty scoring: frequency, recency, duration weighted composite

## Performance
- [ ] Add streaming/incremental computation to `compute_retention` for large viewer datasets
- [ ] Implement approximate quantile computation (t-digest) for percentile metrics at scale
- [ ] Use integer arithmetic for `assign_variant` FNV-1a hash instead of string operations
- [ ] Add batch processing for `analyze_session` across multiple sessions in parallel
- [x] Implement reservoir sampling for memory-bounded attention heatmap generation

## Testing
- [ ] Test `assign_variant` distribution uniformity with chi-squared test over 10K+ assignments
- [ ] Add tests for `winning_variant` with known p-value outcomes (verify statistical correctness)
- [ ] Test `compute_retention` with synthetic viewer data: 100% retention, 50% linear drop, step function
- [ ] Test `drop_off_points` detection accuracy with synthetic retention curves
- [ ] Test `linear_regression_slope` against known regression datasets
- [ ] Add test for `attention_heatmap` with overlapping and non-overlapping session segments

## Documentation
- [ ] Add example workflow: create experiment -> assign users -> collect metrics -> determine winner
- [ ] Document engagement score formula and weight tuning recommendations
- [ ] Add retention curve interpretation guide (what constitutes good/bad retention)
