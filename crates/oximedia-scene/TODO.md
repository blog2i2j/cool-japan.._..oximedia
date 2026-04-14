# oximedia-scene TODO

## Current Status
- 33 modules (9 subdirectory modules) covering scene classification, object/face/logo detection, activity recognition, shot composition analysis, semantic segmentation, saliency detection, aesthetic scoring, event detection, feature extraction, camera motion, crowd density, depth of field, mood analysis, pacing, storyboard generation, summarization, transitions, visual rhythm
- Common types: Point, Rect (with IoU), Confidence
- All algorithms patent-free (HOG, Haar cascades, color histograms, motion histograms, spectral saliency, graph-based segmentation)
- Dependencies: oximedia-core, oximedia-cv, scirs2-core, rayon, serde, thiserror

## Enhancements
- [ ] Add multi-scale detection in `detect::face::FaceDetector` for improved accuracy across face sizes
- [x] Extend `classify::scene::SceneClassifier` with temporal smoothing to avoid flickering classifications
- [ ] Add `composition::rules::CompositionAnalyzer` support for golden ratio and phi grid in addition to rule of thirds
- [x] Implement non-maximum suppression (NMS) in `detect` for overlapping detections across all detector types
- [ ] Extend `saliency` with temporal saliency for video (motion-weighted attention maps)
- [x] Add `scene_boundary` configurable threshold sensitivity with automatic threshold estimation
- [ ] Improve `camera_motion` estimation with robust RANSAC-based homography fitting
- [ ] Extend `aesthetic` scoring with content-type-specific models (landscape, portrait, action, still life)

## New Features
- [ ] Add `text_detection` module for detecting and localizing text regions in frames (scene text, overlays, subtitles)
- [ ] Implement `emotion_recognition` module analyzing facial expressions in detected faces
- [ ] Add `object_tracking` module for tracking detected objects across frames with Kalman filtering
- [ ] Implement `scene_captioning` module generating natural-language descriptions from scene features
- [ ] Add `content_moderation` module for detecting sensitive content categories (violence, NSFW)
- [ ] Implement `temporal_graph` module connecting scene analysis results across time for narrative structure
- [ ] Add `thumbnail_selector` module choosing the most visually representative frame per scene
- [ ] Implement `motion_energy` module for quantifying overall motion intensity per scene segment
- [ ] Add `audio_visual_correlation` module detecting sync between audio events and visual changes

## Performance
- [ ] Add rayon parallel processing in `segment::Segmenter` for graph-based segmentation on large frames
- [ ] Implement multi-resolution pyramid processing in `detect` to avoid full-resolution scans
- [ ] Add batch frame processing in `classify` to amortize model initialization across frames
- [ ] Cache feature descriptors in `features` module to avoid recomputation when multiple analyzers use them
- [ ] Optimize `saliency::spectral` FFT-based saliency with pre-allocated FFT buffers
- [ ] Add frame decimation in `summarization` to process every Nth frame for long videos

## Testing
- [ ] Add accuracy benchmarks for `detect::face` using standard face detection datasets (FDDB-like metrics)
- [ ] Test `scene_boundary` detection against known scene-change timestamps in reference videos
- [ ] Add tests for `composition::rules` with synthetically generated images with known compositions
- [ ] Test `classification` consistency across sequential frames of the same scene
- [ ] Add regression tests for `aesthetic` scoring to ensure score stability across versions

## Documentation
- [ ] Document all patent-free algorithms used with academic references and complexity analysis
- [ ] Add guide for combining multiple analysis modules into a complete video analysis pipeline
- [ ] Document confidence threshold selection guidelines for each detector type
- [ ] Add performance benchmarks table (frames/second) for each analysis module at common resolutions
