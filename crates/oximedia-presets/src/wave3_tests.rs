//! Wave 3 additional tests for oximedia-presets.
//!
//! Covers:
//! - Lazy-load preset categories
//! - Global cached PresetLibrary
//! - Platform preset validation (YouTube, Twitch, ATSC)
//! - AbrLadder tests for HLS and DASH
//! - OptimalPreset edge cases (zero bitrate, u64::MAX bitrate)
//! - Export/Import round-trip tests
//! - preset_diff additional tests

use crate::{
    broadcast, platform,
    preset_diff::{DiffEntry, PresetDiff, PresetDiffCompare},
    streaming, LazyPresetCategory, OptimalPreset, Preset, PresetCategory, PresetLibrary,
    PresetMetadata,
};

// ── Lazy-load preset category tests ───────────────────────────────────────────

#[test]
fn test_lazy_hls_not_loaded_initially() {
    // Create a fresh LazyPresetCategory and verify the initial state machine.
    let cat = LazyPresetCategory::new("HLS-test", streaming::hls::all_presets);
    assert!(
        !cat.is_loaded(),
        "Fresh LazyPresetCategory should not be loaded yet"
    );
}

#[test]
fn test_lazy_category_loads_on_first_get() {
    let cat = LazyPresetCategory::new("HLS-get", streaming::hls::all_presets);
    assert!(!cat.is_loaded(), "Should not be loaded before first get");
    let presets = cat.get();
    assert!(cat.is_loaded(), "Should be loaded after first get");
    assert!(
        !presets.is_empty(),
        "Loaded HLS presets should be non-empty"
    );
}

#[test]
fn test_lazy_category_same_slice_on_repeated_get() {
    let cat = LazyPresetCategory::new("HLS-repeat", streaming::hls::all_presets);
    let first = cat.get();
    let second = cat.get();
    // Both slices must point to the same allocation (OnceLock guarantees this).
    assert_eq!(
        first.as_ptr(),
        second.as_ptr(),
        "Repeated get() must return the same allocation"
    );
}

#[test]
fn test_lazy_category_isolation() {
    // Accessing hls_cat should not trigger loading of yt_cat.
    let hls_cat = LazyPresetCategory::new("HLS-iso", streaming::hls::all_presets);
    let yt_cat = LazyPresetCategory::new("YT-iso", platform::youtube::all_presets);
    assert!(!hls_cat.is_loaded());
    assert!(!yt_cat.is_loaded());
    let _ = hls_cat.get();
    assert!(hls_cat.is_loaded(), "HLS should be loaded after get");
    assert!(
        !yt_cat.is_loaded(),
        "YouTube should NOT be loaded when only HLS was accessed"
    );
}

#[test]
fn test_lazy_category_name() {
    let cat = LazyPresetCategory::new("my-cat", streaming::hls::all_presets);
    assert_eq!(cat.name(), "my-cat");
}

#[test]
fn test_lazy_category_contains_correct_presets() {
    let cat = LazyPresetCategory::new("YT-check", platform::youtube::all_presets);
    let presets = cat.get();
    // All returned presets should have the youtube tag.
    for p in presets {
        assert!(
            p.has_tag("youtube"),
            "YouTube lazy category should only return youtube-tagged presets; '{}' missing tag",
            p.metadata.id
        );
    }
}

#[test]
fn test_lazy_broadcast_category_loads() {
    let cat = LazyPresetCategory::new("ATSC", broadcast::atsc::all_presets);
    assert!(!cat.is_loaded());
    let presets = cat.get();
    assert!(cat.is_loaded());
    assert!(
        !presets.is_empty(),
        "ATSC broadcast presets must be non-empty"
    );
    for p in presets {
        assert!(
            p.has_tag("broadcast") || p.has_tag("atsc"),
            "ATSC preset '{}' must have broadcast or atsc tag",
            p.metadata.id
        );
    }
}

// ── Global library cache tests ─────────────────────────────────────────────────

#[test]
fn test_global_library_same_pointer() {
    // Call global() 100 times and verify the raw pointer never changes.
    let first_ptr = std::ptr::from_ref(PresetLibrary::global());
    for _ in 0..100 {
        let ptr = std::ptr::from_ref(PresetLibrary::global());
        assert_eq!(
            first_ptr, ptr,
            "global() must return the same address every call"
        );
    }
}

#[test]
fn test_global_library_has_presets() {
    let lib = PresetLibrary::global();
    assert!(
        lib.count() > 100,
        "Global library should contain 100+ presets, got {}",
        lib.count()
    );
}

#[test]
fn test_global_library_contains_youtube() {
    let lib = PresetLibrary::global();
    let yt = lib.find_by_category(PresetCategory::Platform("YouTube".to_string()));
    assert!(
        !yt.is_empty(),
        "Global library should contain YouTube presets"
    );
}

#[test]
fn test_global_library_contains_hls() {
    let lib = PresetLibrary::global();
    let hls = lib.find_by_tag("hls");
    assert!(
        !hls.is_empty(),
        "Global library should contain HLS-tagged presets"
    );
}

// ── Platform preset validation tests ────────────────────────────────────────────

#[test]
fn test_youtube_1080p_preset_validation() {
    let library = PresetLibrary::new();
    // Find a YouTube 1080p H.264 preset.
    let preset = library.get("youtube-1080p").or_else(|| {
        library
            .find_by_category(PresetCategory::Platform("YouTube".to_string()))
            .into_iter()
            .find(|p| p.has_tag("1080p") && !p.has_tag("hdr") && !p.has_tag("60fps"))
    });
    let p = preset.expect("YouTube 1080p preset should exist");
    // Codec must be h264 or av1.
    let codec = p.config.video_codec.as_deref().unwrap_or("");
    assert!(
        codec == "h264" || codec == "av1",
        "YouTube 1080p codec must be h264 or av1, got '{codec}'"
    );
    // Bitrate in [4_000_000, 20_000_000] bps.
    let bitrate = p.config.video_bitrate.unwrap_or(0);
    assert!(
        (4_000_000..=20_000_000).contains(&bitrate),
        "YouTube 1080p bitrate {bitrate} bps out of expected [4M, 20M] range"
    );
    // Resolution must be 1920×1080.
    assert_eq!(
        p.config.width,
        Some(1920),
        "YouTube 1080p width must be 1920"
    );
    assert_eq!(
        p.config.height,
        Some(1080),
        "YouTube 1080p height must be 1080"
    );
}

#[test]
fn test_twitch_low_latency_preset_max_bitrate() {
    let library = PresetLibrary::new();
    let presets = library.find_by_category(PresetCategory::Platform("Twitch".to_string()));
    assert!(!presets.is_empty(), "Twitch presets must exist");
    // All Twitch presets must have video bitrate ≤ 8_500_000 bps (8.5 Mbps source quality max).
    for p in &presets {
        if let Some(vbr) = p.config.video_bitrate {
            assert!(
                vbr <= 8_500_000,
                "Twitch preset '{}' video bitrate {}bps exceeds max 8.5 Mbps",
                p.metadata.id,
                vbr
            );
        }
    }
}

#[test]
fn test_twitch_presets_audio_bitrate_minimum() {
    let library = PresetLibrary::new();
    let presets = library.find_by_category(PresetCategory::Platform("Twitch".to_string()));
    for p in &presets {
        if let Some(abr) = p.config.audio_bitrate {
            assert!(
                abr >= 128_000,
                "Twitch preset '{}' audio bitrate {}bps below 128 kbps minimum",
                p.metadata.id,
                abr
            );
        }
    }
}

#[test]
fn test_atsc_broadcast_video_bitrate_minimum() {
    let library = PresetLibrary::new();
    let atsc_presets = library.find_by_category(PresetCategory::Broadcast("ATSC".to_string()));
    assert!(
        !atsc_presets.is_empty(),
        "ATSC presets must exist in library"
    );
    for p in &atsc_presets {
        let vbr = p.config.video_bitrate.unwrap_or(0);
        assert!(
            vbr >= 10_000_000,
            "ATSC preset '{}' video bitrate {}bps below 10 Mbps minimum",
            p.metadata.id,
            vbr
        );
    }
}

#[test]
fn test_atsc_broadcast_audio_codec_present() {
    let library = PresetLibrary::new();
    let atsc_presets = library.find_by_category(PresetCategory::Broadcast("ATSC".to_string()));
    for p in &atsc_presets {
        let audio = p.config.audio_codec.as_deref().unwrap_or("");
        assert!(
            !audio.is_empty(),
            "ATSC preset '{}' must have an audio codec configured",
            p.metadata.id
        );
    }
}

// ── AbrLadder tests ───────────────────────────────────────────────────────────

#[test]
fn test_hls_abr_ladder_rung_count() {
    let ladder = streaming::hls::hls_abr_ladder();
    assert!(
        ladder.rungs.len() >= 3,
        "HLS ABR ladder must have at least 3 rungs, got {}",
        ladder.rungs.len()
    );
}

#[test]
fn test_hls_abr_ladder_distinct_resolutions() {
    let ladder = streaming::hls::hls_abr_ladder();
    let heights: Vec<u32> = ladder.rungs.iter().map(|r| r.height).collect();
    let mut unique_heights = heights.clone();
    unique_heights.dedup();
    assert_eq!(
        heights.len(),
        unique_heights.len(),
        "HLS ladder rungs must have distinct heights (no duplicates)"
    );
}

#[test]
fn test_hls_abr_ladder_bitrate_monotonically_increasing() {
    let ladder = streaming::hls::hls_abr_ladder();
    for w in ladder.rungs.windows(2) {
        assert!(
            w[1].bitrate > w[0].bitrate,
            "HLS ladder bitrates must be strictly increasing: rung with height {} has {}bps, next {} has {}bps",
            w[0].height, w[0].bitrate, w[1].height, w[1].bitrate
        );
    }
}

#[test]
fn test_dash_abr_ladder_rung_count() {
    let ladder = streaming::dash::dash_abr_ladder();
    assert!(
        ladder.rungs.len() >= 3,
        "DASH ABR ladder must have at least 3 rungs, got {}",
        ladder.rungs.len()
    );
}

#[test]
fn test_dash_abr_ladder_bitrate_monotonically_increasing() {
    let ladder = streaming::dash::dash_abr_ladder();
    for w in ladder.rungs.windows(2) {
        assert!(
            w[1].bitrate > w[0].bitrate,
            "DASH ladder bitrates must be strictly increasing"
        );
    }
}

#[test]
fn test_rtmp_presets_have_rtmp_tag() {
    let library = PresetLibrary::new();
    let rtmp_presets = library.find_by_tag("rtmp");
    assert!(
        !rtmp_presets.is_empty(),
        "RTMP presets must exist in library"
    );
    for p in &rtmp_presets {
        assert!(
            p.has_tag("rtmp"),
            "Preset '{}' returned by find_by_tag('rtmp') must have rtmp tag",
            p.metadata.id
        );
    }
}

// ── OptimalPreset edge-case tests ─────────────────────────────────────────────

#[test]
fn test_optimal_preset_zero_bitrate_returns_lowest() {
    let library = PresetLibrary::new();
    // Zero bitrate: all presets exceed target → returns the lowest-bitrate preset.
    let preset = OptimalPreset::select(&library, 0);
    assert!(
        preset.is_some(),
        "select() with bitrate=0 must return the lowest quality preset, not None"
    );
    let returned_bitrate = preset
        .and_then(|p| p.config.video_bitrate)
        .unwrap_or(u64::MAX);
    let min_bitrate = library
        .presets_iter()
        .filter_map(|p| p.config.video_bitrate)
        .min()
        .unwrap_or(0);
    assert_eq!(
        returned_bitrate, min_bitrate,
        "select(0) must return the preset with minimum video bitrate"
    );
}

#[test]
fn test_optimal_preset_max_bitrate_returns_highest() {
    let library = PresetLibrary::new();
    let preset = OptimalPreset::select(&library, u64::MAX);
    assert!(
        preset.is_some(),
        "select() with bitrate=u64::MAX must return a preset"
    );
    let returned_bitrate = preset.and_then(|p| p.config.video_bitrate).unwrap_or(0);
    let max_bitrate = library
        .presets_iter()
        .filter_map(|p| p.config.video_bitrate)
        .max()
        .unwrap_or(0);
    assert_eq!(
        returned_bitrate, max_bitrate,
        "select(u64::MAX) must return the preset with maximum video bitrate"
    );
}

#[test]
fn test_optimal_preset_protocol_zero_bitrate_no_panic() {
    let library = PresetLibrary::new();
    // Zero bitrate with known protocol must not panic; returns lowest hls preset.
    let result = OptimalPreset::select_for_protocol(&library, 0, "hls");
    assert!(
        result.is_some(),
        "select_for_protocol(0, 'hls') must return Some (lowest HLS preset)"
    );
}

#[test]
fn test_optimal_preset_protocol_max_bitrate_no_panic() {
    let library = PresetLibrary::new();
    let result = OptimalPreset::select_for_protocol(&library, u64::MAX, "hls");
    assert!(
        result.is_some(),
        "select_for_protocol(u64::MAX, 'hls') must return Some (highest HLS preset)"
    );
}

#[test]
fn test_optimal_preset_unknown_protocol_returns_none() {
    let library = PresetLibrary::new();
    // A completely unknown protocol tag has no matching presets → None (no panic).
    let result = OptimalPreset::select_for_protocol(&library, 5_000_000, "zzznoproto");
    assert!(
        result.is_none(),
        "Unknown protocol should return None, not panic"
    );
}

#[test]
fn test_optimal_preset_select_reasonable_bitrate() {
    let library = PresetLibrary::new();
    let result = OptimalPreset::select(&library, 5_000_000);
    assert!(
        result.is_some(),
        "5 Mbps target must find at least one preset"
    );
}

// ── Export / Import round-trip tests ──────────────────────────────────────────

#[test]
fn test_export_import_round_trip_fields() {
    use crate::export::json as export_json;
    use crate::import::json as import_json;

    let metadata = PresetMetadata::new(
        "round-trip-test",
        "Round Trip Test Preset",
        PresetCategory::Custom,
    )
    .with_description("A preset for round-trip testing")
    .with_tag("test")
    .with_tag("roundtrip");

    let preset = Preset::new(metadata, oximedia_transcode::PresetConfig::default());

    let json = export_json::export_to_string(&preset).expect("export_to_string must succeed");
    assert!(!json.is_empty(), "Exported JSON must not be empty");

    let imported = import_json::import_from_string(&json).expect("import_from_string must succeed");

    assert_eq!(imported.metadata.id, preset.metadata.id);
    assert_eq!(imported.metadata.name, preset.metadata.name);
    assert_eq!(imported.metadata.description, preset.metadata.description);
    assert_eq!(imported.metadata.tags, preset.metadata.tags);
}

#[test]
fn test_export_import_idempotent() {
    use crate::export::json as export_json;
    use crate::import::json as import_json;

    let metadata = PresetMetadata::new("idem-test", "Idempotent Test", PresetCategory::Custom)
        .with_tag("idempotent");
    let preset = Preset::new(metadata, oximedia_transcode::PresetConfig::default());

    let json1 = export_json::export_to_string(&preset).expect("first export must succeed");
    let imported = import_json::import_from_string(&json1).expect("import must succeed");
    let json2 = export_json::export_to_string(&imported).expect("second export must succeed");

    assert_eq!(json1, json2, "export→import→export must be idempotent");
}

#[test]
fn test_export_multiple_and_import_multiple() {
    use crate::export::json as export_json;
    use crate::import::json as import_json;

    let presets = vec![
        Preset::new(
            PresetMetadata::new("multi-a", "Multi A", PresetCategory::Custom).with_tag("multi"),
            oximedia_transcode::PresetConfig::default(),
        ),
        Preset::new(
            PresetMetadata::new("multi-b", "Multi B", PresetCategory::Custom).with_tag("multi"),
            oximedia_transcode::PresetConfig::default(),
        ),
    ];

    let json =
        export_json::export_multiple_to_string(&presets).expect("export_multiple must succeed");
    let imported =
        import_json::import_multiple_from_string(&json).expect("import_multiple must succeed");

    assert_eq!(imported.len(), 2, "Must import exactly 2 presets");
    assert_eq!(imported[0].metadata.id, "multi-a");
    assert_eq!(imported[1].metadata.id, "multi-b");
}

#[test]
fn test_import_invalid_json_returns_err() {
    use crate::import::json as import_json;
    let result = import_json::import_from_string("{ invalid json {{ }}");
    assert!(
        result.is_err(),
        "Importing invalid JSON must return Err, not panic"
    );
}

#[test]
fn test_export_import_preserves_category() {
    use crate::export::json as export_json;
    use crate::import::json as import_json;

    let metadata = PresetMetadata::new(
        "cat-test",
        "Category Test",
        PresetCategory::Streaming("HLS".to_string()),
    );
    let preset = Preset::new(metadata, oximedia_transcode::PresetConfig::default());
    let json = export_json::export_to_string(&preset).expect("export must succeed");
    let imported = import_json::import_from_string(&json).expect("import must succeed");
    assert_eq!(
        imported.metadata.category,
        PresetCategory::Streaming("HLS".to_string()),
        "Category must survive JSON round-trip"
    );
}

#[test]
fn test_export_import_preserves_author() {
    use crate::export::json as export_json;
    use crate::import::json as import_json;

    let mut metadata = PresetMetadata::new("author-test", "Author Test", PresetCategory::Custom);
    metadata.author = "TestAuthor".to_string();
    let preset = Preset::new(metadata, oximedia_transcode::PresetConfig::default());
    let json = export_json::export_to_string(&preset).expect("export must succeed");
    let imported = import_json::import_from_string(&json).expect("import must succeed");
    assert_eq!(
        imported.metadata.author, "TestAuthor",
        "Author field must survive JSON round-trip"
    );
}

// ── preset_diff additional tests ──────────────────────────────────────────────

#[test]
fn test_preset_diff_identical_empty_map() {
    use std::collections::HashMap;
    let map: HashMap<String, String> = HashMap::new();
    let diff = PresetDiffCompare::compare(&map, &map);
    assert!(
        diff.is_empty(),
        "Comparing identical empty maps must produce empty diff"
    );
}

#[test]
fn test_preset_diff_single_field_change() {
    use std::collections::HashMap;
    let mut base = HashMap::new();
    base.insert("video_codec".to_string(), "h264".to_string());
    base.insert("bitrate".to_string(), "5000000".to_string());
    let mut modified = base.clone();
    modified.insert("bitrate".to_string(), "8000000".to_string());
    let diff = PresetDiffCompare::compare(&base, &modified);
    assert_eq!(diff.len(), 1, "Only one field changed (bitrate)");
    assert_eq!(diff.entries()[0].field, "bitrate");
    assert_eq!(diff.entries()[0].before, "5000000");
    assert_eq!(diff.entries()[0].after, "8000000");
}

#[test]
fn test_preset_diff_all_fields_changed() {
    use std::collections::HashMap;
    let mut base = HashMap::new();
    base.insert("a".to_string(), "1".to_string());
    base.insert("b".to_string(), "2".to_string());
    base.insert("c".to_string(), "3".to_string());
    let mut modified = HashMap::new();
    modified.insert("a".to_string(), "10".to_string());
    modified.insert("b".to_string(), "20".to_string());
    modified.insert("c".to_string(), "30".to_string());
    let diff = PresetDiffCompare::compare(&base, &modified);
    assert_eq!(diff.len(), 3, "All three fields changed");
}

#[test]
fn test_preset_diff_has_breaking_codec_change() {
    let mut diff = PresetDiff::new();
    diff.push(DiffEntry::new("video_codec", "h264", "av1", true));
    assert!(
        PresetDiffCompare::has_breaking_changes(&diff),
        "Codec change must be reported as breaking"
    );
}

#[test]
fn test_preset_diff_description_change_not_breaking() {
    let mut diff = PresetDiff::new();
    diff.push(DiffEntry::new("description", "old desc", "new desc", false));
    assert!(
        !PresetDiffCompare::has_breaking_changes(&diff),
        "Description change must NOT be reported as breaking"
    );
}

#[test]
fn test_preset_diff_changed_fields_list() {
    let mut diff = PresetDiff::new();
    diff.push(DiffEntry::new("width", "1280", "1920", true));
    diff.push(DiffEntry::new("height", "720", "1080", true));
    let fields = diff.changed_fields();
    assert!(
        fields.iter().any(|e| e.field == "width"),
        "Changed fields must include 'width'"
    );
    assert!(
        fields.iter().any(|e| e.field == "height"),
        "Changed fields must include 'height'"
    );
}
