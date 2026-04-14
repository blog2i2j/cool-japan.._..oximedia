# oximedia-timesync TODO

## Current Status
- 67 source files across 46+ public modules providing comprehensive time synchronization
- PTP (IEEE 1588-2019): clock (discipline, drift, holdover, offset, selection), ptp (clock, message, port, bmca, slave, transparent, dataset), boundary_clock, gptp
- NTP (RFC 5905): ntp (client, packet, pool, filter, stratum)
- Timecode sync: timecode (mod, ltc, mtc, smpte, jam_sync)
- Clock discipline: clock_discipline, clock_domain, clock_ensemble, clock_error, clock_recovery, clock_steering
- Monitoring: drift_monitor, sync_audit, sync_metrics, sync_monitor, sync_stats, sync_status, sync_window
- Media sync: sync (genlock, audio, video), aes67, dante_clock, frequency_estimator, frequency_sync, holdover_estimator, jitter_buffer, leap_second, offset_correction, offset_filter, phase_lock, reference_clock, sync_protocol, time_reference
- IPC: ipc (socket, shmem), ffi (clock_adjust)
- Integration: integration module with timestamp conversion utilities
- Dependencies: oximedia-core, oximedia-timecode, bytes, bitflags, chrono, crc, tokio, memmap2, libc (non-wasm)

## Enhancements
- [ ] Add PTP Announce message timeout handling in `ptp::bmca` for detecting master clock loss and triggering re-election
- [ ] Extend `ntp::client` to support NTS (Network Time Security, RFC 8915) for authenticated time synchronization
- [ ] Improve `clock::drift::DriftEstimator` with Allan variance computation for oscillator characterization
- [ ] Add `clock_ensemble` support for weighted averaging of multiple clock sources using Bayesian estimation
- [ ] Extend `aes67` module with PTP profile compliance checking per AES67-2013 specification
- [ ] Improve `jitter_buffer` with adaptive depth adjustment based on measured network jitter statistics
- [ ] Add `sync_audit` persistent logging to file with rotation for post-production timing analysis
- [ ] Extend `gptp` (gPTP/802.1AS) with neighbor rate ratio measurement for improved accuracy

## New Features
- [x] Implement `white_rabbit` module for White Rabbit sub-nanosecond PTP extension (used in broadcast facilities)
- [ ] Add `smpte_2059` module for SMPTE ST 2059 PTP profile for professional media (media-specific PTP profile)
- [x] Implement `clock_graph` module for visualizing clock hierarchy and sync relationships in multi-device setups
- [ ] Add `ravenna` module for RAVENNA AoIP clock synchronization profile support
- [ ] Implement `sync_test` module with synthetic clock drift/offset injection for testing sync algorithms
- [ ] Add `gps_reference` module for GPS-disciplined clock input as ultimate reference source
- [ ] Implement `ptp_management` module for PTP management messages (GET/SET/COMMAND) per IEEE 1588
- [x] Add `clock_quality_monitor` that tracks and reports MTIE (Maximum Time Interval Error) and TDEV

## Performance
- [ ] Use lock-free shared memory updates in `ipc::shmem` for microsecond-level timestamp distribution
- [ ] Implement batched PTP message processing in `ptp::port` to handle burst arrival of sync/follow-up messages
- [ ] Add hardware timestamping support detection in `ptp::clock` for sub-microsecond PTP accuracy
- [ ] Use SIMD-accelerated CRC computation in PTP message validation
- [ ] Implement zero-allocation PTP message parsing in `ptp::message` using nom or manual byte parsing

## Testing
- [ ] Add PTP BMCA election test with multiple clock candidates at different priorities and verify correct grandmaster selection
- [ ] Test `clock::holdover` accuracy degradation over time with known oscillator drift model
- [ ] Add NTP client test with simulated server responses at various stratum levels and verify correct server selection
- [ ] Test `timecode::jam_sync` lock acquisition and holdover behavior with intermittent LTC signal
- [ ] Add `genlock` test verifying frame-edge alignment within +/-1 sample at 48kHz for genlock output
- [ ] Test `leap_second` handling at UTC midnight boundary with PTP and NTP sources
- [ ] Add `dante_clock` interop test verifying compatibility with Dante clock domain behavior

## Documentation
- [ ] Add clock hierarchy diagram showing PTP grandmaster -> boundary clock -> ordinary clock -> application
- [ ] Document sync accuracy expectations for each protocol (PTP: <1us, NTP: <10ms, LTC: 1 frame, genlock: sub-sample)
- [ ] Add deployment guide for broadcast facility time synchronization with PTP, genlock, and LTC
