# oximedia-net TODO

## Current Status
- 26 source files/directories covering major streaming protocols
- Protocols: HLS, DASH, RTMP, SRT, WebRTC, SMPTE ST 2110, QUIC
- Infrastructure: CDN (multi-CDN failover, circuit breaker), ABR (adaptive bitrate), FEC (forward error correction)
- Network: connection_pool, bandwidth_estimator, bandwidth_throttle, flow_control, packet_buffer, qos_monitor
- Utilities: protocol_detect, retry_policy, session_tracker, stream_mux, multicast, network_path, rtp_session
- Re-exports: SRT stats, encryption session, ABR controller/variant/bandwidth types

## Enhancements
- [ ] Implement CMAF low-latency HLS (LL-HLS) with partial segment support in hls module
- [ ] Add DASH low-latency (LL-DASH) with chunked transfer encoding in dash module
- [ ] Implement SRT caller/listener/rendezvous connection modes fully in srt module
- [ ] Add RTMP enhanced mode (RTMP+) with AV1/VP9 codec support in rtmp module
- [ ] Implement WebRTC WHIP/WHEP signaling for browser-based ingest/playback in webrtc module
- [x] Add QUIC datagram mode for ultra-low-latency media transport in quic module
- [x] Improve ABR algorithm with buffer-based (BBA) strategy alongside bandwidth-based in abr module
- [x] Add SRT encryption with AES-256-GCM in addition to current AES-128-CTR

## New Features
- [ ] Implement RIST (Reliable Internet Stream Transport) protocol as alternative to SRT
- [x] Add Zixi-compatible protocol support for broadcast contribution links
- [ ] Implement SMPTE ST 2022-7 seamless protection switching (dual-path redundancy) in smpte2110
- [ ] Add media relay/restreaming server -- receive from one protocol, retransmit on another
- [ ] Implement bandwidth-aware transcoding trigger -- signal codec crate to reduce quality when bandwidth drops
- [ ] Add ICE (Interactive Connectivity Establishment) for WebRTC NAT traversal
- [ ] Implement multipath streaming -- send redundant streams over multiple network interfaces

## Performance
- [ ] Use connection pooling in CDN module for keep-alive HTTP connections to edge servers
- [ ] Implement zero-copy segment serving using sendfile/splice syscalls in HLS/DASH server
- [ ] Add io_uring support for high-throughput RTP packet handling in smpte2110
- [ ] Implement packet pacing in SRT to smooth out burst traffic and reduce jitter
- [ ] Profile and optimize FEC encoding/decoding -- consider SIMD-accelerated XOR operations
- [ ] Cache parsed manifest/playlist structures to avoid re-parsing on each client request

## Testing
- [ ] Add HLS playlist generation test verifying M3U8 format compliance (EXT-X-VERSION, segment tags)
- [ ] Test DASH MPD generation against schema validation
- [ ] Add SRT connection test with simulated packet loss verifying ARQ retransmission
- [ ] Test CDN failover: simulate primary CDN timeout, verify automatic switch to secondary
- [ ] Add ABR test: simulate fluctuating bandwidth, verify smooth quality transitions without rebuffering
- [ ] Test RTMP handshake and chunk stream parsing against reference implementation output

## Documentation
- [ ] Document supported protocol matrix with capabilities (latency, reliability, encryption, ABR)
- [ ] Add protocol selection guide: when to use HLS vs DASH vs SRT vs WebRTC vs ST 2110
- [ ] Document CDN configuration with examples for Cloudflare, Fastly, CloudFront integration
