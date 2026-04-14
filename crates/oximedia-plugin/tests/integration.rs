//! Integration test: full plugin lifecycle (register → lookup → use → unregister).
//!
//! Tests the end-to-end workflow that a host application would follow:
//! 1. Create a `PluginRegistry`.
//! 2. Register one or more plugins.
//! 3. Look up codecs and verify availability.
//! 4. Attempt to create decoders/encoders.
//! 5. Unregister all plugins and verify the registry is empty.

use oximedia_codec::{CodecError, EncoderConfig};
use oximedia_plugin::{
    CodecPluginInfo, PluginCapability, PluginRegistry, StaticPlugin, PLUGIN_API_VERSION,
};
use std::collections::HashMap;
use std::sync::Arc;

fn make_plugin(name: &str, codecs: &[(&str, bool, bool)]) -> Arc<dyn oximedia_plugin::CodecPlugin> {
    let info = CodecPluginInfo {
        name: name.to_string(),
        version: "1.0.0".to_string(),
        author: "Integration Test".to_string(),
        description: format!("Integration test plugin: {name}"),
        api_version: PLUGIN_API_VERSION,
        license: "MIT".to_string(),
        patent_encumbered: false,
    };

    let mut plugin = StaticPlugin::new(info);
    for (codec, decode, encode) in codecs {
        plugin = plugin.add_capability(PluginCapability {
            codec_name: (*codec).to_string(),
            can_decode: *decode,
            can_encode: *encode,
            pixel_formats: vec!["yuv420p".to_string()],
            properties: HashMap::new(),
        });
    }
    Arc::new(plugin)
}

/// Full lifecycle: register → lookup → create → clear.
#[test]
fn test_full_plugin_lifecycle() {
    // 1. Create registry.
    let registry = PluginRegistry::empty();
    assert_eq!(registry.plugin_count(), 0);

    // 2. Register plugins.
    let av1_plugin = make_plugin("av1-plugin", &[("av1", true, true)]);
    let vp9_plugin = make_plugin("vp9-plugin", &[("vp9", true, true), ("vp8", true, false)]);
    registry.register(av1_plugin).expect("register av1");
    registry.register(vp9_plugin).expect("register vp9");
    assert_eq!(registry.plugin_count(), 2);

    // 3. Verify codec availability.
    assert!(registry.has_codec("av1"));
    assert!(registry.has_codec("vp9"));
    assert!(registry.has_codec("vp8"));
    assert!(!registry.has_codec("h264"));

    assert!(registry.has_decoder("av1"));
    assert!(registry.has_encoder("av1"));
    assert!(registry.has_decoder("vp8"));
    assert!(!registry.has_encoder("vp8")); // vp8 is decode-only

    // 4. Plugin listing.
    let plugins = registry.list_plugins();
    assert_eq!(plugins.len(), 2);
    let names: Vec<&str> = plugins.iter().map(|p| p.name.as_str()).collect();
    assert!(names.contains(&"av1-plugin"));
    assert!(names.contains(&"vp9-plugin"));

    // 5. Codec listing.
    let codecs = registry.list_codecs();
    assert_eq!(codecs.len(), 3);

    // 6. find_plugin_for_codec returns correct plugin.
    let av1_info = registry.find_plugin_for_codec("av1").expect("find av1");
    assert_eq!(av1_info.name, "av1-plugin");

    let vp8_info = registry.find_plugin_for_codec("vp8").expect("find vp8");
    assert_eq!(vp8_info.name, "vp9-plugin");

    // 7. Create decoder (fails because StaticPlugin has no decoder factory).
    let decode_result = registry.find_decoder("av1");
    assert!(decode_result.is_err()); // No factory registered, expected

    // 8. Create encoder (fails for same reason).
    let encode_result = registry.find_encoder("av1", EncoderConfig::default());
    assert!(encode_result.is_err());

    // 9. Unregister all plugins.
    registry.clear();
    assert_eq!(registry.plugin_count(), 0);
    assert!(!registry.has_codec("av1"));
    assert!(registry.list_plugins().is_empty());
    assert!(registry.list_codecs().is_empty());
}

/// Duplicate plugin registration is rejected.
#[test]
fn test_duplicate_registration_rejected() {
    let registry = PluginRegistry::empty();
    let p1 = make_plugin("my-plugin", &[("h264", true, false)]);
    let p2 = make_plugin("my-plugin", &[("h265", true, false)]);
    registry.register(p1).expect("first registration");
    let err = registry.register(p2).expect_err("second should fail");
    assert!(err.to_string().contains("already registered"));
}

/// Wrong API version is rejected.
#[test]
fn test_wrong_api_version_rejected() {
    let registry = PluginRegistry::empty();
    let info = CodecPluginInfo {
        name: "bad-plugin".to_string(),
        version: "1.0.0".to_string(),
        author: "Test".to_string(),
        description: "Bad API version".to_string(),
        api_version: 999,
        license: "MIT".to_string(),
        patent_encumbered: false,
    };
    let p = Arc::new(StaticPlugin::new(info));
    let err = registry.register(p).expect_err("should fail");
    assert!(err.to_string().contains("API"));
}

/// find_decoder returns error for unknown codec.
#[test]
fn test_find_decoder_unknown_codec() {
    let registry = PluginRegistry::empty();
    registry
        .register(make_plugin("p", &[("vp9", true, true)]))
        .expect("register");
    let result = registry.find_decoder("h264");
    assert!(result.is_err());
    let err = result.err().expect("err");
    assert!(err.to_string().contains("h264"));
}

/// find_encoder returns error for unknown codec.
#[test]
fn test_find_encoder_unknown_codec() {
    let registry = PluginRegistry::empty();
    registry
        .register(make_plugin("p", &[("vp9", true, true)]))
        .expect("register");
    let result = registry.find_encoder("h265", EncoderConfig::default());
    assert!(result.is_err());
    let err = result.err().expect("err");
    assert!(err.to_string().contains("h265"));
}

/// Plugin with factory produces correct error behaviour.
#[test]
fn test_plugin_with_decoder_factory() {
    let info = CodecPluginInfo {
        name: "factory-plugin".to_string(),
        version: "1.0.0".to_string(),
        author: "Test".to_string(),
        description: "Plugin with decoder factory".to_string(),
        api_version: PLUGIN_API_VERSION,
        license: "MIT".to_string(),
        patent_encumbered: false,
    };

    let plugin = StaticPlugin::new(info)
        .add_capability(PluginCapability {
            codec_name: "test".to_string(),
            can_decode: true,
            can_encode: false,
            pixel_formats: vec![],
            properties: HashMap::new(),
        })
        .with_decoder(|codec_name| {
            Err(CodecError::UnsupportedFeature(format!(
                "Mock: decoder not available for '{codec_name}'"
            )))
        });

    let registry = PluginRegistry::empty();
    registry.register(Arc::new(plugin)).expect("register");

    assert!(registry.has_decoder("test"));
    let result = registry.find_decoder("test");
    assert!(result.is_err(), "factory must return error");
    let err = result.err().expect("error");
    assert!(err.to_string().contains("Mock"));
}

// ── Extended plugin lifecycle integration tests ───────────────────────────────

/// Priority ordering: higher-priority plugin wins codec lookup.
#[test]
fn test_lifecycle_priority_ordering_determines_codec_winner() {
    let registry = PluginRegistry::empty();
    let low = make_plugin("low-pri-plugin", &[("opus", true, true)]);
    let high = make_plugin("high-pri-plugin", &[("opus", true, true)]);
    registry
        .register_with_priority(low, 0)
        .expect("register low");
    registry
        .register_with_priority(high, 100)
        .expect("register high");
    let found = registry.find_plugin_for_codec("opus").expect("codec found");
    assert_eq!(
        found.name, "high-pri-plugin",
        "highest priority plugin must win"
    );
}

/// Unregister one plugin — other plugin\'s codecs remain available.
#[test]
fn test_lifecycle_unregister_one_codec_remains_through_other() {
    let registry = PluginRegistry::empty();
    registry
        .register(make_plugin("flac-plugin", &[("flac", true, true)]))
        .expect("register flac");
    registry
        .register(make_plugin("opus-plugin", &[("opus", true, true)]))
        .expect("register opus");
    registry.unregister("flac-plugin").expect("unregister flac");
    assert!(!registry.has_codec("flac"), "flac must be gone");
    assert!(registry.has_codec("opus"), "opus must still be available");
}

/// Unregister-then-reregister: codec becomes available again.
#[test]
fn test_lifecycle_unregister_then_reregister() {
    let registry = PluginRegistry::empty();
    registry
        .register(make_plugin("vorbis-plugin", &[("vorbis", true, false)]))
        .expect("register first");
    assert!(registry.has_codec("vorbis"));
    registry.unregister("vorbis-plugin").expect("unregister");
    assert!(!registry.has_codec("vorbis"));
    registry
        .register(make_plugin("vorbis-plugin", &[("vorbis", true, true)]))
        .expect("re-register");
    assert!(registry.has_codec("vorbis"));
    assert!(registry.has_encoder("vorbis"), "encoder now present");
}

/// find_plugin_for_codec returns None for unknown codec in empty registry.
#[test]
fn test_lifecycle_find_plugin_empty_registry() {
    let registry = PluginRegistry::empty();
    assert!(registry.find_plugin_for_codec("h264").is_none());
    assert!(registry.find_plugin_for_codec("vp9").is_none());
}

/// A single plugin providing multiple codecs exposes all of them.
#[test]
fn test_lifecycle_multi_codec_plugin_all_codecs_accessible() {
    let registry = PluginRegistry::empty();
    let multi = make_plugin(
        "multi-codec",
        &[
            ("av1", true, true),
            ("vp9", true, true),
            ("vp8", true, false),
            ("theora", true, false),
        ],
    );
    registry.register(multi).expect("register multi");
    assert_eq!(registry.list_codecs().len(), 4);
    for codec in &["av1", "vp9", "vp8", "theora"] {
        assert!(
            registry.has_codec(codec),
            "codec {codec} should be available"
        );
    }
    assert!(registry.has_encoder("av1"));
    assert!(!registry.has_encoder("vp8"), "vp8 is decode-only");
}

/// Unregistering the primary codec provider makes the fallback take over.
#[test]
fn test_lifecycle_codec_failover_after_unregister() {
    let registry = PluginRegistry::empty();
    registry
        .register(make_plugin("primary-h264", &[("h264", true, true)]))
        .expect("register primary");
    registry
        .register(make_plugin("fallback-h264", &[("h264", true, false)]))
        .expect("register fallback");
    let winner = registry.find_plugin_for_codec("h264").expect("found");
    assert_eq!(winner.name, "primary-h264");
    registry
        .unregister("primary-h264")
        .expect("unregister primary");
    let winner2 = registry.find_plugin_for_codec("h264").expect("fallback");
    assert_eq!(winner2.name, "fallback-h264");
}

/// clear() removes all codecs from the capability cache.
#[test]
fn test_lifecycle_clear_invalidates_all_capability_lookups() {
    let registry = PluginRegistry::empty();
    registry
        .register(make_plugin("codec-a", &[("av1", true, true)]))
        .expect("a");
    registry
        .register(make_plugin("codec-b", &[("vp9", true, false)]))
        .expect("b");
    registry.clear();
    assert!(!registry.has_codec("av1"), "av1 cleared");
    assert!(!registry.has_codec("vp9"), "vp9 cleared");
    assert_eq!(registry.plugin_count(), 0);
}

/// Negative priority plugins are registered but never preferred.
#[test]
fn test_lifecycle_negative_priority_plugin_still_registered() {
    let registry = PluginRegistry::empty();
    registry
        .register_with_priority(
            make_plugin("fallback-only", &[("theora", true, false)]),
            -100,
        )
        .expect("negative priority");
    assert_eq!(registry.plugin_count(), 1);
    assert!(registry.has_codec("theora"));
    assert_eq!(registry.plugin_priority("fallback-only"), Some(-100));
}

/// Unregistering a non-existent plugin returns an informative error.
#[test]
fn test_lifecycle_unregister_nonexistent_returns_not_found() {
    let registry = PluginRegistry::empty();
    registry
        .register(make_plugin("real-plugin", &[("vp8", true, false)]))
        .expect("register");
    let err = registry
        .unregister("ghost-plugin")
        .expect_err("should fail");
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("not found") || err_msg.contains("ghost-plugin"),
        "error must reference missing plugin, got: {err_msg}"
    );
}

/// All registered plugins appear in list_plugins.
#[test]
fn test_lifecycle_list_plugins_contains_all_registered() {
    let registry = PluginRegistry::empty();
    let names = ["alpha", "beta", "gamma", "delta"];
    for name in &names {
        registry.register(make_plugin(name, &[])).expect("register");
    }
    let listed: Vec<String> = registry
        .list_plugins()
        .into_iter()
        .map(|i| i.name)
        .collect();
    assert_eq!(listed.len(), names.len());
    for name in &names {
        assert!(listed.contains(&name.to_string()), "missing {name}");
    }
}
