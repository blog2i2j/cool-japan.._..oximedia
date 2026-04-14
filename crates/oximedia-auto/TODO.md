# oximedia-auto TODO

## Current Status
- 55 modules providing automated video editing: highlights, cuts, assembly, rules, scoring, pacing curves, subject tracking, reframing, and more
- Core modules: `smart_crop`, `smart_reframe` (+ `SubjectTracker`, `VerticalToHorizontalParams`), `smart_trim`, `music_sync`, `tempo_detect`, `narrative`, `scene_classifier`, `color_match`, `subtitle_sync`, `tag_suggest`, `visual_theme`
- Pacing: `pacing_curve` (`PacingCurve` with 7 `CurveShape` variants, `CurveAnalyser`, `CurveStats`)
- Complete `AutoEditor` pipeline with `auto_edit()` orchestrating detect-score-cut-assemble workflow
- All numerical operations use plain Rust primitives (f32/f64/Vec/fixed arrays); no ndarray dependency

## Enhancements
- [x] Replace `ndarray` dependency with SciRS2-Core per SCIRS2 policy (confirmed: ndarray was never present in Cargo.toml or source; all array ops use plain Rust)
- [ ] Improve `highlights::HighlightDetector` with configurable multi-pass analysis (coarse then fine)
- [ ] Add confidence scores to `cuts::CutPoint` for user review prioritization
- [ ] Extend `assembly::AssemblyType` with `Recap` variant for episode/series recaps
- [x] Add `rules::PacingPreset::Custom` with user-defined shot duration curves (`pacing_curve` module: `PacingCurve`, `CurveShape` 7 variants, `CurveKeyframe`, `CurveAnalyser`; `distribute_clips`/`compute_cut_positions`; 27 tests)
- [ ] Improve `scoring::SceneScorer` with temporal context (score relative to neighbors)
- [x] Extend `smart_reframe` with subject tracking across multiple frames for smooth panning (`SubjectTracker` + `SubjectBounds` with EMA; `generate_sequence`/`generate_smooth_sequence`; 10 tests)
- [x] Add vertical-to-horizontal reframing in `smart_reframe` (`VerticalToHorizontalParams`, `VerticalToHorizontalStrategy` 5 variants, `FrameOrientation`; `primary_placement`/`side_regions`/`saliency_crop_window`; 7 tests)
- [ ] Improve `music_sync` with downbeat vs upbeat distinction for edit point selection

## New Features
- [ ] Add `auto_thumbnail` module for automatic thumbnail selection from best frames
- [ ] Implement `auto_chaptering` module to generate chapter points from scene analysis
- [ ] Add `content_warning` module for automatic content classification (violence, language)
- [ ] Implement `engagement_predictor` module using interest curve analysis for audience retention
- [ ] Add `a_b_roll` module for automatic B-roll insertion suggestions based on dialogue content
- [ ] Implement `color_continuity` checker that flags jarring color shifts between assembled clips
- [ ] Add platform-specific export presets for YouTube Shorts, Instagram Reels, TikTok in `assembly`

## Performance
- [ ] Parallelize `highlights::detect_highlights()` across frame batches using rayon
- [ ] Cache scene features in `scoring::SceneScorer` to avoid recomputation on config changes
- [ ] Add early termination to `cuts::detect_cuts()` when sufficient cut points found for target duration
- [ ] Use downscaled frames for initial `smart_crop` pass, refine on full resolution only for final crops
- [ ] Implement lazy evaluation for `AutoEditResult` fields that may not be needed by caller

## Testing
- [ ] Add test for `auto_edit()` end-to-end with synthetic video frames and audio
- [ ] Test `rules::RulesEngine::apply_rules()` enforces minimum shot duration constraints
- [ ] Add regression test for `assembly::generate_social_clip()` ensuring output within duration tolerance
- [ ] Test `smart_trim` preserves dialogue boundaries when trimming
- [ ] Add benchmark comparing `scoring` performance across different `ScoringConfig` settings
- [ ] Test `color_match` produces consistent results for identical input pairs

## Documentation
- [ ] Document use-case presets ("trailer", "highlights", "social") with expected behavior
- [ ] Add flowchart for `auto_edit()` pipeline showing data flow between stages
- [ ] Document `scoring::FeatureWeights` tuning guidelines for different content types
