//! Matroska/WebM parser fuzzer.
//!
//! This fuzzer tests the Matroska demuxer for:
//! - EBML header parsing
//! - Segment element parsing
//! - Cluster and block parsing
//! - Lacing algorithms (EBML, Xiph, fixed)
//! - Edge cases in VINT encoding
//! - Malformed inputs
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use bytes::Bytes;
use oximedia_container::demux::{Demuxer, MatroskaDemuxer};
use oximedia_io::MemorySource;

fuzz_target!(|data: &[u8]| {
    // Create a memory source from the fuzzer input
    let source = MemorySource::new(Bytes::copy_from_slice(data));
    let mut demuxer = MatroskaDemuxer::new(source);

    // Create a minimal runtime for async operations
    let rt = match tokio::runtime::Builder::new_current_thread().build() {
        Ok(rt) => rt,
        Err(_) => return,
    };

    rt.block_on(async {
        // Try to probe the format
        // This should never panic, even on malformed input
        let _ = demuxer.probe().await;

        // Try to read packets
        // Limit iterations to prevent infinite loops on malformed data
        for _ in 0..1000 {
            match demuxer.read_packet().await {
                Ok(_packet) => {
                    // Successfully read a packet
                    // Continue reading
                }
                Err(_) => {
                    // Error or EOF - stop reading
                    break;
                }
            }
        }

        // Check streams info
        let _ = demuxer.streams();
    });
});
