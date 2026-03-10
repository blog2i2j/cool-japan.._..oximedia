//! WAV/RIFF parser fuzzer.
//!
//! This fuzzer tests the WAV demuxer for:
//! - RIFF header parsing
//! - fmt chunk parsing
//! - data chunk extraction
//! - Extended format codes (WAVE_FORMAT_EXTENSIBLE)
//! - Edge cases in PCM configurations
//! - Malformed chunk structures
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use bytes::Bytes;
use oximedia_container::demux::{Demuxer, WavDemuxer};
use oximedia_io::MemorySource;

fuzz_target!(|data: &[u8]| {
    // Create a memory source from the fuzzer input
    let source = MemorySource::new(Bytes::copy_from_slice(data));
    let mut demuxer = WavDemuxer::new(source);

    // Create a minimal runtime for async operations
    let rt = match tokio::runtime::Builder::new_current_thread().build() {
        Ok(rt) => rt,
        Err(_) => return,
    };

    rt.block_on(async {
        // Try to probe the format
        let _ = demuxer.probe().await;

        // Try to read packets
        // Limit iterations to prevent infinite loops
        for _ in 0..1000 {
            match demuxer.read_packet().await {
                Ok(_packet) => {
                    // Successfully read a packet
                }
                Err(_) => {
                    // Error or EOF
                    break;
                }
            }
        }

        // Check streams
        let _ = demuxer.streams();
    });
});
