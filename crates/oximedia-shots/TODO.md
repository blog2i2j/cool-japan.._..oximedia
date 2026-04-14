# oximedia-shots TODO

## Current Status
- 38 public modules + lib.rs (~70 source files total) covering shot detection, classification, camera movement, composition analysis, continuity checking, pattern analysis, and export
- Key features: cut/dissolve/fade/wipe detection, 7 shot types (ECU to ELS), camera angle/movement classification, scene detection, coverage analysis, EDL/CSV/JSON export
- Dependencies: oximedia-core, oximedia-cv, oximedia-scene, oximedia-edl, oximedia-timecode

## Enhancements
- [x] Add adaptive threshold tuning in `detect::CutDetector` based on content complexity (e.g., action vs. dialogue scenes)
- [x] Extend `classify::ShotTypeClassifier` to support insert shots and cutaway classification
- [ ] Add confidence calibration to `classify::AngleClassifier` using histogram-based features
- [x] Improve `composition::CompositionAnalyzer` with golden ratio and phi grid analysis in addition to rule of thirds
- [ ] Add multi-threaded frame processing in `ShotDetector::detect_shots` using rayon parallel iterators
- [ ] Extend `export::shotlist` to support XML-based shot list formats (FCP XML, Resolve markers)
- [ ] Add configurable wipe direction patterns in `detect::WipeDetector` (radial, iris, clock)
- [ ] Improve `continuity::ContinuityChecker` to detect 180-degree rule violations using optical flow direction

## New Features
- [ ] Add ML-based shot boundary detection module using lightweight convolutional features (no external ML runtime)
- [ ] Implement `shot_similarity` module for finding visually similar shots across a project using perceptual hashing
- [ ] Add `color_continuity` module to detect color temperature/grading inconsistencies between consecutive shots
- [ ] Implement `depth_of_field` estimation module to classify shallow vs. deep focus shots
- [ ] Add `audio_scene_boundary` integration with audio energy envelope for scene boundary refinement
- [ ] Implement real-time shot detection mode in `ShotDetector` with streaming frame input
- [ ] Add `shot_annotation` module for attaching free-form metadata and tags to detected shots

## Performance
- [ ] Cache histogram computations in `detect::CutDetector` to avoid recomputing for overlapping frame pairs
- [ ] Use downscaled proxy frames (e.g., 160x90) in `camera::MovementDetector` for faster optical flow estimation
- [ ] Implement block-based SAD comparison in `detect::DissolveDetector` instead of full-frame pixel difference
- [ ] Add early termination in `classify::ShotTypeClassifier` when confidence exceeds 0.95 threshold
- [ ] Pool `FrameBuffer` allocations in `detect_shots` to reduce allocation pressure on long sequences

## Testing
- [ ] Add integration tests for `detect_shots` with synthetic frame sequences containing known cuts and dissolves
- [ ] Test `continuity::ContinuityChecker` with crafted jump-cut and axis-crossing scenarios
- [ ] Add round-trip tests for `export::edl` and `export::shotlist` (export then re-import)
- [ ] Test `scene::SceneDetector` with multi-scene sequences having distinct visual characteristics
- [ ] Add property-based tests for `pattern::RhythmAnalyzer` with varying shot duration distributions

## Documentation
- [ ] Add architecture diagram showing the full detection pipeline (tracking -> estimation -> classification -> export)
- [ ] Document threshold tuning guidelines for different content types (documentary, action, interview)
- [ ] Add inline examples to `ShotDetector::detect_shots` showing how to iterate results
