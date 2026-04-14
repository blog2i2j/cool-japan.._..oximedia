# oximedia-automation TODO

## Current Status
- 37 modules providing broadcast automation: master control, channel automation, device control, playlist execution, live switching, failover, EAS, as-run logging, monitoring, remote control, Lua scripting
- Hierarchical architecture: MasterControl -> ChannelAutomation -> PlayoutEngine/DeviceController/LiveSwitcher
- Sub-directories: `channel/`, `device/`, `eas/`, `failover/`, `live/`, `logging/`, `master/`, `monitor/`, `playlist/`, `protocol/`, `remote/`, `script/`
- Uses `mlua` for Lua54 scripting, `axum` for remote API, `tokio-serial` for RS-422

## Enhancements
- [x] Add `automation_log` module to lib.rs exports (file exists at `src/automation_log.rs` but not declared)
- [x] Add `cue_trigger` module to lib.rs exports (file exists at `src/cue_trigger.rs` but not declared)
- [x] Add `interlock` module to lib.rs exports (file exists at `src/interlock.rs` but not declared)
- [x] Extend `eas::alert` with CAP (Common Alerting Protocol) XML parsing support
- [x] Add SCTE-35 splice insert/return command generation in `playlist`
- [x] Implement playlist item pre-roll verification in `channel::playout` to catch missing media before air
- [x] Extend `failover::manager` with N+1 redundancy mode (not just hot standby pairs)
- [x] Add GPI trigger debouncing logic in `device_control` to prevent false triggers
- [x] Implement `script::engine` sandboxing with resource limits (memory, CPU time) for Lua scripts
- [x] Implement playlist item pre-roll verification (`playlist::preroll::MediaVerifier` with magic-byte format detection)

## New Features
- [x] Add `graphics_overlay` module for automated CG/lower-third insertion during playout
- [x] Implement `ad_insertion` module for SCTE-35 based dynamic ad break management
- [x] Add `multi_site` module for coordinating automation across geographically distributed facilities
- [x] Implement `timecode_sync` module for house sync/genlock timecode distribution
- [x] Add `regulatory_compliance` module for automated content rating insertion per jurisdiction
- [x] Implement `equipment_inventory` module tracking device serial numbers, firmware versions, maintenance schedules
- [x] Add WebSocket real-time event streaming in `remote` for dashboard live updates

## Performance
- [x] Use lock-free channels for `event_bus` instead of mutex-guarded queues
- [x] Implement connection pooling for `remote::server` HTTP sessions
- [x] Add batched as-run log writes in `logging::asrun` to reduce I/O frequency
- [x] Cache compiled Lua scripts in `script::engine` to avoid re-parsing on repeated execution
- [x] Use arena allocation for `playlist` item storage to reduce allocation overhead during playout

## Testing
- [x] Add integration test for complete playout cycle: load playlist, start, verify as-run log entries
- [x] Test `failover::manager` automatic switchover under simulated primary failure
- [x] Add EAS alert insertion test verifying correct audio/video interruption and restoration
- [x] Test `device_control` VDCP and Sony 9-pin protocol message serialization round-trips
- [x] Add stress test for `master::control` handling 50+ simultaneous channel automations
- [x] Test `script::engine` Lua sandbox prevents filesystem and network access
- [x] Test `script::engine` resource limits (instruction count, timeout, normal completion)
- [x] Test `playlist::preroll::MediaVerifier` with real temp files (MP4, MKV, WAV, FLAC, MXF)

## Documentation
- [ ] Document supported device control protocols with message format specifications
- [ ] Add example Lua automation scripts for common workflows (break insertion, logo switching)
- [ ] Document EAS compliance requirements and how the system meets them
- [ ] Add network topology diagram for multi-channel remote control setup
