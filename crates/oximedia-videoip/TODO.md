# oximedia-videoip TODO

## Current Status
- 44 source files implementing professional video-over-IP protocol (patent-free NDI alternative)
- Key features: low-latency UDP transport with FEC, mDNS/DNS-SD service discovery, VP9/AV1/v210/UYVY video codecs, Opus/PCM audio formats, tally lights, PTZ control, timecode, metadata, jitter buffering, packet loss recovery, multi-stream support
- Modules: bandwidth_est, bonding, codec, color_space_conv, congestion, discovery, encryption, fec, flow_monitor, flow_stats, frame_pacing, jitter, metadata, multicast, multicast_group, ndi_bridge, nmos, packet, packet_loss, ptp_boundary, ptz, quic_transport, receiver, redundancy, rist, sdp, smpte2110, source, srt_config, stats, stream_descriptor, stream_health, stream_recorder, stream_sync, tally, transport, utils (bandwidth, connection, quality, ring_buffer)
- Dependencies: oximedia-core, oximedia-codec, oximedia-net, oximedia-monitor, oximedia-timecode, tokio, socket2, mdns-sd, reed-solomon-erasure, crossbeam, flume

## Enhancements
- [ ] Implement actual QUIC transport in `quic_transport` using quinn or equivalent pure-Rust QUIC library
- [x] Add adaptive FEC rate in `fec` module that adjusts parity packet ratio based on measured `packet_loss` rate
- [x] Extend `congestion` control with BBR-style algorithm instead of basic AIMD for better bandwidth utilization
- [ ] Implement actual PTP (IEEE 1588) clock synchronization in `ptp_boundary` with sub-microsecond accuracy
- [ ] Add NMOS IS-04/IS-05 full registration and connection management in `nmos` module
- [ ] Extend `encryption` module with DTLS-SRTP for standard-compliant media encryption
- [x] Improve `jitter` buffer with adaptive depth based on network conditions (expand during congestion, shrink during stability)
- [ ] Add actual SDP offer/answer negotiation in `sdp` module for standard SIP/WebRTC interop

## New Features
- [ ] Implement `whip_whep` module for WHIP/WHEP protocol support (WebRTC-based ingest/egress)
- [ ] Add `srt_transport` module implementing full SRT (Secure Reliable Transport) protocol in pure Rust
- [ ] Implement `rtsp_server` module for RTSP/RTP source serving to standard media players
- [ ] Add `stream_recording_mux` that records incoming streams directly to MKV/WebM containers
- [ ] Implement `bandwidth_shaping` module for traffic shaping and QoS prioritization per stream
- [ ] Add `multiview` module for combining multiple receiver streams into a single mosaic output
- [ ] Implement `stream_relay` for re-broadcasting received streams to multiple downstream receivers
- [ ] Add `diagnostic_overlay` module that burns network stats (latency, packet loss, bitrate) onto video frames

## Performance
- [ ] Optimize `fec` Reed-Solomon encoding/decoding with SIMD-accelerated Galois field arithmetic
- [ ] Implement zero-copy packet path in `transport` using `bytes::Bytes` throughout without intermediate copies
- [ ] Add lock-free ring buffer in `utils::ring_buffer` for audio/video frame handoff between network and processing threads
- [ ] Optimize `color_space_conv` with SIMD for v210-to-planar and UYVY-to-planar conversion hot paths
- [ ] Implement scatter/gather I/O (sendmmsg/recvmmsg) in UDP transport for reduced syscall overhead
- [ ] Profile and optimize `frame_pacing` to use precise timer (mach_absolute_time on macOS) instead of sleep-based pacing

## Testing
- [ ] Add network simulation tests for `congestion` control under varying latency/loss conditions
- [ ] Test `fec` recovery with configurable packet loss patterns (burst, random, periodic)
- [ ] Add integration test for full source->receiver round-trip with loopback UDP
- [ ] Test `discovery` mDNS announcement and resolution with multiple concurrent sources
- [ ] Benchmark `jitter` buffer at various network jitter levels (1ms, 5ms, 20ms, 50ms)
- [ ] Test `stream_sync` lip-sync accuracy between audio and video streams under packet reordering

## Documentation
- [ ] Document protocol wire format specification (packet header layout, control message types)
- [ ] Add network configuration guide (firewall ports, multicast setup, bandwidth requirements per resolution)
- [ ] Document SMPTE 2110 compatibility mapping between VideoIP types and ST 2110-20/30/40
