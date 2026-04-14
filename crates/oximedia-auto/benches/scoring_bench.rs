use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_auto::scoring::{
    FeatureWeights, SceneFeatures, SceneScorer, ScoringConfig, TemporalContextConfig,
};
use oximedia_core::{Rational, Timestamp};
use std::hint::black_box;

/// Build a `SceneFeatures` value from explicit field values for benchmarking.
#[allow(clippy::too_many_arguments)]
fn make_features(
    motion: f64,
    face_count: usize,
    face_coverage: f64,
    audio_peak: f64,
    audio_energy: f64,
    color_diversity: f64,
    edge_density: f64,
    brightness_mean: f64,
    contrast: f64,
    sharpness: f64,
    object_count: usize,
    object_diversity: f64,
    temporal_stability: f64,
) -> SceneFeatures {
    SceneFeatures {
        motion_intensity: motion,
        face_count,
        face_coverage,
        audio_peak,
        audio_energy,
        color_diversity,
        edge_density,
        brightness_mean,
        contrast,
        sharpness,
        object_count,
        object_diversity,
        temporal_stability,
    }
}

/// A small set of representative `ScoringConfig` presets used as benchmark variants.
fn make_configs() -> Vec<(&'static str, ScoringConfig)> {
    // --- Default / General Highlights ---
    let default_cfg = ScoringConfig::default();

    // --- Sports / Action ---
    let sports_weights = FeatureWeights {
        motion: 2.5,
        face: 0.8,
        audio_peak: 2.0,
        audio_energy: 1.5,
        color: 0.5,
        edge: 0.6,
        contrast: 0.4,
        sharpness: 0.6,
        object: 0.6,
    };
    let sports_cfg = ScoringConfig::default().with_feature_weights(sports_weights);

    // --- Interview / Documentary ---
    let interview_weights = FeatureWeights {
        motion: 0.5,
        face: 2.5,
        audio_peak: 1.5,
        audio_energy: 0.8,
        color: 0.6,
        edge: 0.5,
        contrast: 0.5,
        sharpness: 1.0,
        object: 0.8,
    };
    let interview_cfg = ScoringConfig::default().with_feature_weights(interview_weights);

    // --- Music Video ---
    let music_weights = FeatureWeights {
        motion: 1.8,
        face: 1.0,
        audio_peak: 2.2,
        audio_energy: 1.8,
        color: 1.2,
        edge: 0.8,
        contrast: 0.8,
        sharpness: 0.6,
        object: 0.5,
    };
    let music_cfg = ScoringConfig::default().with_feature_weights(music_weights);

    // --- Nature / B-roll ---
    let nature_weights = FeatureWeights {
        motion: 0.6,
        face: 0.3,
        audio_peak: 0.5,
        audio_energy: 0.4,
        color: 2.0,
        edge: 1.2,
        contrast: 1.5,
        sharpness: 1.8,
        object: 1.0,
    };
    let nature_cfg = ScoringConfig::default().with_feature_weights(nature_weights);

    // --- Classification disabled (faster path) ---
    let no_classify_cfg = ScoringConfig::default()
        .with_classification(false)
        .with_sentiment(false)
        .with_auto_titling(false);

    vec![
        ("default", default_cfg),
        ("sports_action", sports_cfg),
        ("interview", interview_cfg),
        ("music_video", music_cfg),
        ("nature_broll", nature_cfg),
        ("no_classification", no_classify_cfg),
    ]
}

/// A representative set of `SceneFeatures` covering low / mid / high activity.
fn make_scene_batch() -> Vec<SceneFeatures> {
    vec![
        // Low activity (static establishing shot)
        make_features(
            0.05, 0, 0.0, 0.1, 0.08, 0.3, 0.2, 0.6, 0.25, 0.7, 1, 0.2, 0.95,
        ),
        // Mid activity (dialogue, two faces)
        make_features(
            0.25, 2, 0.35, 0.6, 0.55, 0.5, 0.4, 0.55, 0.5, 0.75, 3, 0.4, 0.8,
        ),
        // High activity (action / sports)
        make_features(
            0.85, 1, 0.15, 0.9, 0.88, 0.65, 0.7, 0.5, 0.7, 0.6, 6, 0.7, 0.45,
        ),
        // Close-up face
        make_features(
            0.12, 1, 0.62, 0.45, 0.4, 0.35, 0.3, 0.58, 0.45, 0.82, 2, 0.3, 0.88,
        ),
        // Music-heavy (high audio, colourful, moderate motion)
        make_features(
            0.55, 0, 0.0, 0.95, 0.92, 0.85, 0.6, 0.52, 0.68, 0.55, 4, 0.65, 0.6,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Benchmark 1: `score_scene` across all config presets for each sample scene
// ---------------------------------------------------------------------------
fn bench_score_scene_configs(c: &mut Criterion) {
    let configs = make_configs();
    let scenes = make_scene_batch();

    let timebase = Rational::new(1, 1000);
    let start = Timestamp::new(0, timebase);
    let end = Timestamp::new(2000, timebase);

    let mut group = c.benchmark_group("score_scene/configs");

    for (config_name, config) in &configs {
        let scorer = SceneScorer::new(config.clone());

        group.bench_with_input(
            BenchmarkId::new("config", config_name),
            config_name,
            |b, _| {
                b.iter(|| {
                    for features in &scenes {
                        let result = scorer
                            .score_scene(
                                black_box(start),
                                black_box(end),
                                black_box(features.clone()),
                            )
                            .ok();
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2: composite_score() directly on FeatureWeights (no allocation)
// ---------------------------------------------------------------------------
fn bench_composite_score(c: &mut Criterion) {
    let scenes = make_scene_batch();
    let configs = make_configs();

    let mut group = c.benchmark_group("composite_score/weights");

    for (config_name, config) in &configs {
        let weights = config.feature_weights.clone();

        group.bench_with_input(
            BenchmarkId::new("weights", config_name),
            config_name,
            |b, _| {
                b.iter(|| {
                    let mut total = 0.0f64;
                    for features in &scenes {
                        total += features.composite_score(black_box(&weights));
                    }
                    black_box(total)
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3: temporal context scoring (neighbor-aware path)
// ---------------------------------------------------------------------------
fn bench_score_scene_with_context(c: &mut Criterion) {
    let configs = make_configs();
    let scenes = make_scene_batch();

    let timebase = Rational::new(1, 1000);
    let start = Timestamp::new(0, timebase);
    let end = Timestamp::new(2000, timebase);
    let neighbor_scores: Vec<f64> = vec![0.4, 0.55, 0.62, 0.48, 0.70];

    let mut group = c.benchmark_group("score_scene/temporal_context");

    for (config_name, config) in &configs {
        let scorer = SceneScorer::new(config.clone())
            .with_temporal_context(TemporalContextConfig::default());

        group.bench_with_input(
            BenchmarkId::new("config", config_name),
            config_name,
            |b, _| {
                b.iter(|| {
                    for features in &scenes {
                        let result = scorer
                            .score_scene_with_context(
                                black_box(start),
                                black_box(end),
                                black_box(features.clone()),
                                black_box(&neighbor_scores),
                            )
                            .ok();
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 4: cache warm-up vs. steady-state throughput
// ---------------------------------------------------------------------------
fn bench_cache_warmup_vs_steady(c: &mut Criterion) {
    let scenes = make_scene_batch();
    let timebase = Rational::new(1, 1000);
    let start = Timestamp::new(0, timebase);
    let end = Timestamp::new(2000, timebase);

    let mut group = c.benchmark_group("score_scene/cache");

    // Cold cache — scorer is recreated each iteration so cache starts empty
    group.bench_function("cold_cache", |b| {
        b.iter(|| {
            let scorer = SceneScorer::new(ScoringConfig::default());
            for features in &scenes {
                let result = scorer
                    .score_scene(
                        black_box(start),
                        black_box(end),
                        black_box(features.clone()),
                    )
                    .ok();
                black_box(result);
            }
        });
    });

    // Warm cache — scorer is shared across iterations; first call populates cache
    let warm_scorer = SceneScorer::new(ScoringConfig::default());
    // Pre-populate the cache
    for features in &scenes {
        let _ = warm_scorer.score_scene(start, end, features.clone());
    }

    group.bench_function("warm_cache", |b| {
        b.iter(|| {
            for features in &scenes {
                let result = warm_scorer
                    .score_scene(
                        black_box(start),
                        black_box(end),
                        black_box(features.clone()),
                    )
                    .ok();
                black_box(result);
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 5: batch scoring (many scenes, single config) — throughput test
// ---------------------------------------------------------------------------
fn bench_batch_scoring_throughput(c: &mut Criterion) {
    const BATCH_SIZE: usize = 256;
    let template_scenes = make_scene_batch();

    // Build a large synthetic batch by cycling through the template scenes
    let batch: Vec<SceneFeatures> = (0..BATCH_SIZE)
        .map(|i| template_scenes[i % template_scenes.len()].clone())
        .collect();

    let configs = make_configs();
    let timebase = Rational::new(1, 1000);

    let mut group = c.benchmark_group("batch_scoring/throughput");
    group.throughput(criterion::Throughput::Elements(BATCH_SIZE as u64));

    for (config_name, config) in &configs {
        let scorer = SceneScorer::new(config.clone());

        group.bench_with_input(
            BenchmarkId::new("config", config_name),
            config_name,
            |b, _| {
                let start = Timestamp::new(0, timebase);
                let end = Timestamp::new(500, timebase);

                b.iter(|| {
                    let mut sum = 0.0f64;
                    for features in &batch {
                        if let Ok(scene) = scorer.score_scene(
                            black_box(start),
                            black_box(end),
                            black_box(features.clone()),
                        ) {
                            sum += scene.score;
                        }
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_score_scene_configs,
    bench_composite_score,
    bench_score_scene_with_context,
    bench_cache_warmup_vs_steady,
    bench_batch_scoring_throughput,
);
criterion_main!(benches);
