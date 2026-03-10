//! MP4/ISOBMFF parser fuzzer.
//!
//! This fuzzer tests the MP4 demuxer for:
//! - Box/atom parsing
//! - ftyp detection (rejecting patented codecs)
//! - moov and trak parsing
//! - Sample table (stts, stsc, stco, stsz) parsing
//! - mdat data extraction
//! - Fragmented MP4 (fMP4) support
//! - Edge cases in timing calculations
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use bytes::Bytes;
use oximedia_container::demux::{Demuxer, Mp4Demuxer};
use oximedia_io::MemorySource;

fuzz_target!(|data: &[u8]| {
    // Create a memory source from the fuzzer input
    let source = MemorySource::new(Bytes::copy_from_slice(data));
    let mut demuxer = Mp4Demuxer::new(source);

    // Create a minimal runtime for async operations
    let rt = match tokio::runtime::Builder::new_current_thread().build() {
        Ok(rt) => rt,
        Err(_) => return,
    };

    rt.block_on(async {
        // Try to probe the format
        // This should reject files with patented codecs
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
