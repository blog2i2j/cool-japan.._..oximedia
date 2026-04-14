# oximedia-plugin TODO

## Current Status
- 24 modules: capability, config_persist, config_persistence, error, filter_plugin, graceful_reload, harness, health, health_check, health_monitor, hot_reload, lazy, lazy_init, manifest, plugin_config, plugin_telemetry, pool, priority, registry, resources, sandbox, static_plugin, traits, version_resolver, loader (feature-gated)
- Supports static and dynamic plugin registration via `CodecPlugin` trait
- Features: hot reload with file watching, plugin manifest with dependency resolution, sandboxing with permission sets, version constraint solving
- Filter/transform plugin type (`FilterPlugin` trait + `FilterRegistry`)
- Health check monitoring (`PluginHealthMonitor` with periodic liveness probes, history window, degradation detection)
- Configuration persistence (`PluginConfigStore` — typed JSON-backed key-value store per plugin)
- Feature gate: `dynamic-loading` (libloading) for shared library loading
- 271+ tests passing (160 unit + 33 fuzz_manifest + 9 hot_reload + 16 integration + 37 sandbox + 13 version_graph + 3 doc)

## Enhancements
- [ ] Add plugin priority/ordering in `PluginRegistry` for codec conflict resolution (multiple plugins for same codec)
- [ ] Implement plugin health checks in `hot_reload::HotReloadManager` (periodic liveness probe)
- [ ] Extend `sandbox::PermissionSet` with fine-grained filesystem path restrictions (not just PERM_FILESYSTEM)
- [ ] Add plugin resource usage tracking (memory, CPU time) in `SandboxContext`
- [ ] Implement plugin dependency conflict detection in `version_resolver` (diamond dependency problem)
- [ ] Add graceful degradation in `GracefulReload` — serve from old plugin during new plugin initialization
- [ ] Extend `PluginManifest` with minimum OxiMedia version requirement field

## New Features
- [ ] Add WASM plugin support — load plugins compiled to WebAssembly via wasmtime/wasmer
- [ ] Implement plugin marketplace protocol — discovery, download, and verification of remote plugins
- [ ] Add plugin configuration persistence — save/load plugin settings between sessions
- [ ] Implement plugin telemetry collection — anonymous usage stats for plugin authors
- [ ] Add plugin test harness — standardized testing framework for plugin developers
- [ ] Implement plugin isolation via process sandboxing (run plugin in subprocess with IPC)
- [ ] Add filter/transform plugin type alongside codec plugins (video/audio filter plugins)

## Performance
- [ ] Cache plugin capability lookups in `PluginRegistry` with invalidation on register/unregister
- [ ] Implement lazy plugin initialization — defer codec creation until first use
- [ ] Add plugin instance pooling for codecs that are expensive to initialize
- [x] Optimize `compute_hash` in hot_reload to use memory-mapped I/O for large plugin files — `compute_hash_mmap` with `MMAP_THRESHOLD_BYTES` (4 MiB) page-streaming strategy; 9 new tests

## Testing
- [x] Add integration test for full plugin lifecycle (register -> lookup -> use -> unregister) — 10 new tests in `tests/integration.rs` covering priority ordering, failover, clear, re-registration
- [ ] Test `hot_reload` with simulated file modification events and verify seamless reload
- [ ] Add fuzz testing for `PluginManifest` parsing with malformed JSON/TOML
- [x] Test `sandbox` permission enforcement — verify blocked operations raise `SandboxError` — 13 new tests in `tests/sandbox_test.rs` covering path allow-list, CPU quota, combined enforcement
- [ ] Add tests for `version_resolver` with complex dependency graphs (10+ interdependent plugins)

## Documentation
- [ ] Add plugin development guide with step-by-step shared library plugin creation
- [ ] Document `declare_plugin!` macro usage with complete working example
- [ ] Add security model documentation for sandbox permissions and trust levels
