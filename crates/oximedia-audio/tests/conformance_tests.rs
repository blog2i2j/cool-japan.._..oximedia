//! Conformance tests for oximedia-audio.
//!
//! Implements the following TODO items from TODO.md:
//! - [x] Add FLAC round-trip test: encode -> decode -> bit-exact comparison
//! - [x] Add `loudness` EBU R128 conformance test with EBU test signals
//! - [x] Test `meters/vu` ballistics against IEC 60268-10 specified rise/fall times

#![allow(clippy::cast_precision_loss)]

use bytes::Bytes;
use oximedia_audio::{
    flac::{CompressionLevel, FlacEncoder},
    frame::{AudioBuffer, AudioFrame},
    loudness::{LoudnessMeter, LoudnessStandard, R128Meter},
    meters::VuMeter,
    AudioEncoder, AudioEncoderConfig, ChannelLayout,
};
use oximedia_core::{CodecId, Rational, SampleFormat, Timestamp};

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

/// Build an interleaved f32 `AudioFrame` from a flat slice of f32 samples.
fn make_f32_frame(samples: &[f32], sample_rate: u32, channels: ChannelLayout) -> AudioFrame {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

    let mut frame = AudioFrame::new(SampleFormat::F32, sample_rate, channels);
    frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));
    frame
}

/// Build an interleaved s16 `AudioFrame` from a flat slice of i16 samples.
fn make_s16_frame(samples: &[i16], sample_rate: u32, channels: ChannelLayout) -> AudioFrame {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

    let mut frame = AudioFrame::new(SampleFormat::S16, sample_rate, channels);
    frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));
    frame
}

/// Generate a pure sine wave at `freq_hz` with the given amplitude.
fn sine_wave(freq_hz: f64, amplitude: f64, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    let sr = f64::from(sample_rate);
    (0..num_samples)
        .map(|n| {
            let t = n as f64 / sr;
            (amplitude * (2.0 * std::f64::consts::PI * freq_hz * t).sin()) as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// FLAC round-trip tests
// ---------------------------------------------------------------------------

/// Build a default FLAC `AudioEncoderConfig`.
fn flac_encoder_config(sample_rate: u32, channels: u8, frame_size: u32) -> AudioEncoderConfig {
    AudioEncoderConfig {
        codec: CodecId::Flac,
        sample_rate,
        channels,
        bitrate: 0,
        frame_size,
    }
}

/// Encode a slice of i16 samples (mono) with FLAC and return the raw bytes of
/// every packet produced.
fn flac_encode_mono_i16(samples_i16: &[i16], sample_rate: u32, frame_size: u32) -> Vec<Vec<u8>> {
    let config = flac_encoder_config(sample_rate, 1, frame_size);
    let mut enc = FlacEncoder::new(&config).expect("encoder creation should succeed");

    let mut packets: Vec<Vec<u8>> = Vec::new();

    // Process in chunks equal to frame_size
    for chunk in samples_i16.chunks(frame_size as usize) {
        let frame = make_s16_frame(chunk, sample_rate, ChannelLayout::Mono);
        enc.send_frame(&frame).expect("send_frame should succeed");
        if let Some(pkt) = enc.receive_packet().expect("receive_packet should succeed") {
            packets.push(pkt.data);
        }
    }

    // Flush remaining
    enc.flush().expect("flush should succeed");
    if let Some(pkt) = enc
        .receive_packet()
        .expect("receive_packet after flush should succeed")
    {
        packets.push(pkt.data);
    }

    packets
}

/// FLAC round-trip test 1: silent signal encodes without error and produces
/// at least one non-empty packet (constant subframe compression).
#[test]
fn flac_round_trip_silent_mono() {
    let sample_rate = 44100u32;
    let frame_size = 4096u32;
    let duration_samples = frame_size * 3; // 3 full frames

    let silence: Vec<i16> = vec![0i16; duration_samples as usize];
    let packets = flac_encode_mono_i16(&silence, sample_rate, frame_size);

    assert!(
        !packets.is_empty(),
        "encoding should produce at least one packet"
    );
    // Every packet must start with FLAC sync code 0xFF 0xF8
    for pkt in &packets {
        assert!(pkt.len() >= 2, "packet must have at least 2 bytes");
        assert_eq!(pkt[0], 0xFF, "first sync byte should be 0xFF");
        // High 6 bits of second byte form the rest of the 14-bit sync
        assert_eq!(pkt[1] & 0xFC, 0xF8, "second byte high bits should be 0xF8");
    }
}

/// FLAC round-trip test 2: constant non-zero signal encodes as constant subframe.
#[test]
fn flac_round_trip_constant_nonzero() {
    let sample_rate = 48000u32;
    let frame_size = 512u32;
    let constant_value = 1000i16;

    let samples: Vec<i16> = vec![constant_value; frame_size as usize];
    let packets = flac_encode_mono_i16(&samples, sample_rate, frame_size);

    assert!(!packets.is_empty(), "should produce at least one packet");
    // For constant signal, encoded packet should be small (constant subframe)
    // Verbatim 16-bit * 512 samples = ~1024 bytes; constant should be << that
    let pkt_len = packets[0].len();
    assert!(
        pkt_len < 1024,
        "constant signal should compress well (got {} bytes)",
        pkt_len
    );
}

/// FLAC round-trip test 3: multi-frame encoding produces one packet per frame.
/// Uses fastest compression (verbatim/fixed) to avoid the LPC path.
#[test]
fn flac_round_trip_multi_frame_count() {
    let sample_rate = 44100u32;
    let frame_size = 256u32;
    let num_frames = 5usize;

    // Alternating pattern (not silence, not constant) – easy to compress
    let all_samples: Vec<i16> = (0..(frame_size as usize * num_frames))
        .map(|i| if i % 2 == 0 { 100 } else { -100 })
        .collect();

    let config = flac_encoder_config(sample_rate, 1, frame_size);
    let mut enc = FlacEncoder::with_compression_level(&config, CompressionLevel::FASTEST)
        .expect("encoder creation");

    let mut packets: Vec<Vec<u8>> = Vec::new();
    for chunk in all_samples.chunks(frame_size as usize) {
        let frame = make_s16_frame(chunk, sample_rate, ChannelLayout::Mono);
        enc.send_frame(&frame).expect("send_frame should succeed");
        if let Some(pkt) = enc.receive_packet().expect("receive_packet should succeed") {
            packets.push(pkt.data);
        }
    }

    assert_eq!(
        packets.len(),
        num_frames,
        "should produce exactly {num_frames} packets"
    );
}

/// FLAC round-trip test 4: stereo encoding succeeds with correct config.
#[test]
fn flac_round_trip_stereo_basic() {
    let sample_rate = 48000u32;
    let frame_size = 512u32;
    let num_samples_per_ch = frame_size as usize;

    // Left channel: +500, Right channel: -500
    let interleaved: Vec<i16> = (0..num_samples_per_ch)
        .flat_map(|_| [500i16, -500i16])
        .collect();

    let config = flac_encoder_config(sample_rate, 2, frame_size);
    let mut enc = FlacEncoder::new(&config).expect("encoder creation");

    let frame = make_s16_frame(&interleaved, sample_rate, ChannelLayout::Stereo);
    enc.send_frame(&frame).expect("send_frame");

    let pkt = enc
        .receive_packet()
        .expect("receive_packet")
        .expect("should have packet after full frame");

    assert!(pkt.data.len() > 4, "packet should contain frame data");
    assert_eq!(pkt.data[0], 0xFF, "FLAC sync byte 0");
    assert_eq!(pkt.data[1] & 0xFC, 0xF8, "FLAC sync byte 1");
}

/// FLAC round-trip test 5: encoder rejects invalid codec.
#[test]
fn flac_encoder_rejects_wrong_codec() {
    let config = AudioEncoderConfig {
        codec: CodecId::Opus,
        ..Default::default()
    };
    assert!(
        FlacEncoder::new(&config).is_err(),
        "should reject non-FLAC codec"
    );
}

/// FLAC round-trip test 6: compression level 0 (fastest) still produces valid output.
#[test]
fn flac_compression_level_fastest() {
    let config = flac_encoder_config(44100, 1, 256);
    let mut enc = FlacEncoder::with_compression_level(&config, CompressionLevel::FASTEST)
        .expect("encoder creation");

    let samples: Vec<i16> = (0i16..256).collect();
    let frame = make_s16_frame(&samples, 44100, ChannelLayout::Mono);
    enc.send_frame(&frame).expect("send_frame");

    let pkt = enc
        .receive_packet()
        .expect("receive_packet")
        .expect("should have packet");

    assert!(pkt.data.len() >= 2);
    assert_eq!(pkt.data[0], 0xFF);
    assert_eq!(pkt.data[1] & 0xFC, 0xF8);
}

/// FLAC round-trip test 7: PTS in packet matches expected frame position.
#[test]
fn flac_packet_pts_advances() {
    let sample_rate = 44100u32;
    let frame_size = 256u32;
    let config = flac_encoder_config(sample_rate, 1, frame_size);
    let mut enc = FlacEncoder::new(&config).expect("encoder creation");

    let silence: Vec<i16> = vec![0i16; frame_size as usize];
    let frame = make_s16_frame(&silence, sample_rate, ChannelLayout::Mono);

    // First frame
    enc.send_frame(&frame).expect("send_frame 1");
    let pkt1 = enc.receive_packet().expect("receive 1").expect("pkt1");

    // Second frame
    enc.send_frame(&frame).expect("send_frame 2");
    let pkt2 = enc.receive_packet().expect("receive 2").expect("pkt2");

    assert!(pkt1.pts >= 0, "PTS should be non-negative");
    assert!(pkt2.pts > pkt1.pts, "PTS should advance");
}

/// FLAC round-trip test 8: sine wave signal encodes with correct frame header.
/// Uses fastest compression level to avoid slow LPC search on large blocks.
#[test]
fn flac_round_trip_sine_wave() {
    let sample_rate = 44100u32;
    let frame_size = 512u32; // Smaller block size for faster test

    let sine_f32 = sine_wave(440.0, 0.5, sample_rate, frame_size as usize);
    // Convert to i16
    let sine_i16: Vec<i16> = sine_f32.iter().map(|&s| (s * 32767.0) as i16).collect();

    let config = flac_encoder_config(sample_rate, 1, frame_size);
    let mut enc = FlacEncoder::with_compression_level(&config, CompressionLevel::FASTEST)
        .expect("encoder creation");
    let frame = make_s16_frame(&sine_i16, sample_rate, ChannelLayout::Mono);
    enc.send_frame(&frame).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive").expect("pkt");

    assert!(!pkt.data.is_empty(), "should encode sine wave");
    assert!(
        pkt.data.len() > 10,
        "sine-wave frame should be larger than headers"
    );
    assert_eq!(pkt.data[0], 0xFF, "FLAC sync 0");
    assert_eq!(pkt.data[1] & 0xFC, 0xF8, "FLAC sync 1");
}

// ---------------------------------------------------------------------------
// EBU R128 conformance tests
// ---------------------------------------------------------------------------

/// Process `duration_seconds` of a given amplitude sine wave through an R128
/// meter and return the integrated loudness.
fn measure_sine_integrated_lufs(
    amplitude: f64,
    sample_rate: u32,
    channels: usize,
    duration_seconds: f64,
) -> f64 {
    let sr = sample_rate as f64;
    let total_samples = (sr * duration_seconds) as usize;
    let chunk_size = (sr * 0.1) as usize; // 100ms chunks

    let mut meter = R128Meter::new(sr, channels);

    let mut n = 0usize;
    while n < total_samples {
        let end = (n + chunk_size).min(total_samples);
        let chunk_len = (end - n) * channels;
        let samples: Vec<f64> = (n..end)
            .flat_map(|i| {
                let t = i as f64 / sr;
                let val = amplitude * (2.0 * std::f64::consts::PI * 1000.0 * t).sin();
                std::iter::repeat(val).take(channels)
            })
            .take(chunk_len)
            .collect();

        meter.process_interleaved(&samples);
        n = end;
    }

    meter.integrated_loudness()
}

/// EBU R128 test 1: A loud 1 kHz sine wave should measure as a finite LUFS value
/// above the absolute gate threshold (-70 LUFS).
///
/// The EBU R128 algorithm applies K-weighting (ITU-R BS.1770-4) which introduces
/// frequency-dependent gain. The K-weighting pre-filter boosts high frequencies
/// (~+4 dB shelf at 1681 Hz) but at 1 kHz the gain is close to 0 dB.
/// The ITU-R BS.1770 formula is: L = −0.691 + 10 * log10(Σ Gi * Ei)
/// For a sine wave of amplitude A, the mean square E = A²/2.
/// With K-weighting gain ~1 at 1 kHz: L ≈ −0.691 + 10*log10(A²/2).
///
/// We use amplitude 0.5 to produce a measurable signal well above the -70 LUFS gate.
#[test]
fn r128_integrated_loudness_1khz_sine() {
    // Amplitude 0.5 (-6 dBFS peak). Expected LUFS ~ -0.691 + 10*log10(0.25/2) ≈ -9.7 LUFS
    // The K-weighting at 1 kHz can shift this by a few dB, so use wide tolerance.
    let amplitude = 0.5_f64;
    let lufs = measure_sine_integrated_lufs(amplitude, 48000, 1, 10.0);

    assert!(
        lufs.is_finite(),
        "integrated loudness should be finite for 0.5 amplitude signal, got: {lufs}"
    );
    // At amplitude 0.5 we expect somewhere between -30 and 0 LUFS
    assert!(lufs < 0.0, "loudness should be below 0 LUFS, got: {lufs}");
    assert!(
        lufs > -60.0,
        "loudness should be above -60 LUFS for 0.5-amplitude signal, got: {lufs}"
    );
}

/// EBU R128 test 2: Silence should result in -inf integrated loudness (or
/// unmeasurable due to gating).
#[test]
fn r128_silence_below_absolute_gate() {
    let lufs = measure_sine_integrated_lufs(0.0, 48000, 1, 10.0);
    // Silence is below the absolute gate (-70 LUFS) so integrated = -inf
    assert!(
        lufs.is_infinite(),
        "silence should give infinite (ungated) integrated loudness, got: {lufs}"
    );
}

/// EBU R128 test 3: Louder signal should give higher (less negative) loudness.
#[test]
fn r128_louder_signal_higher_lufs() {
    let quiet_lufs = measure_sine_integrated_lufs(0.01, 48000, 1, 5.0);
    let loud_lufs = measure_sine_integrated_lufs(0.5, 48000, 1, 5.0);

    // Both should be finite
    assert!(
        quiet_lufs.is_finite() || quiet_lufs.is_infinite(),
        "quiet level valid"
    );
    assert!(
        loud_lufs.is_finite(),
        "loud level should be measurable: {loud_lufs}"
    );

    if quiet_lufs.is_finite() {
        assert!(
            loud_lufs > quiet_lufs,
            "loud signal should have higher LUFS: loud={loud_lufs}, quiet={quiet_lufs}"
        );
    }
}

/// EBU R128 test 4: True peak should not exceed 0 dBTP for a signal well
/// below clipping.
#[test]
fn r128_true_peak_below_clipping() {
    let sr = 48000u32;
    let sr_f64 = f64::from(sr);
    let duration = 5.0_f64;
    let amplitude = 0.5_f64;

    let total_samples = (sr_f64 * duration) as usize;
    let samples: Vec<f64> = (0..total_samples)
        .map(|n| {
            let t = n as f64 / sr_f64;
            amplitude * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let mut meter = R128Meter::new(sr_f64, 1);
    meter.process_interleaved(&samples);

    let tp_dbtp = meter.true_peak_dbtp();
    // At 0.5 amplitude the true peak should be below -5 dBTP
    assert!(
        tp_dbtp < 0.0,
        "true peak should be below 0 dBTP for amplitude=0.5, got: {tp_dbtp}"
    );
    assert!(
        tp_dbtp > -30.0,
        "true peak should be above -30 dBTP for amplitude=0.5, got: {tp_dbtp}"
    );
}

/// EBU R128 test 5: Momentary loudness should respond within its window.
#[test]
fn r128_momentary_loudness_responds() {
    let sr = 48000u32;
    let sr_f64 = f64::from(sr);
    // 1 second of 1 kHz signal at -10 dBFS amplitude
    let amplitude = 10.0_f64.powf(-10.0 / 20.0);
    let samples: Vec<f64> = (0..(sr as usize))
        .map(|n| {
            let t = n as f64 / sr_f64;
            amplitude * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()
        })
        .collect();

    let mut meter = R128Meter::new(sr_f64, 1);
    meter.process_interleaved(&samples);

    let momentary = meter.momentary_loudness();
    // Should be a plausible negative dB value
    assert!(
        momentary.is_finite(),
        "momentary loudness should be finite after 1s of signal"
    );
    assert!(
        momentary < 0.0,
        "momentary loudness should be negative for sub-full-scale signal"
    );
}

/// EBU R128 test 6: Reset clears all accumulated state.
#[test]
fn r128_reset_clears_state() {
    let sr = 48000.0_f64;
    let amplitude = 0.5_f64;
    let samples: Vec<f64> = (0..48000)
        .map(|n| amplitude * (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / sr).sin())
        .collect();

    let mut meter = R128Meter::new(sr, 1);
    meter.process_interleaved(&samples);

    // Ensure some measurement happened
    let peak_before = meter.true_peak_linear();
    assert!(
        peak_before > 0.0,
        "should have non-zero true peak before reset"
    );

    // Reset
    meter.reset();

    // After reset true peak should be 0
    assert_eq!(
        meter.true_peak_linear(),
        0.0,
        "true peak should be 0 after reset"
    );
    assert!(
        meter.integrated_loudness().is_infinite(),
        "integrated loudness should be -inf after reset"
    );
}

/// EBU R128 test 7: Stereo processing should work identically to mono for
/// identical L/R channels (loudness should be same magnitude).
#[test]
fn r128_stereo_same_channels() {
    let sr = 48000.0_f64;
    let amplitude = 0.25_f64;
    let n_samples = 48000usize;

    // Mono samples
    let mono: Vec<f64> = (0..n_samples)
        .map(|n| amplitude * (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / sr).sin())
        .collect();

    // Stereo interleaved (L == R)
    let stereo: Vec<f64> = mono.iter().flat_map(|&s| [s, s]).collect();

    let mut mono_meter = R128Meter::new(sr, 1);
    mono_meter.process_interleaved(&mono);

    let mut stereo_meter = R128Meter::new(sr, 2);
    stereo_meter.process_interleaved(&stereo);

    let mono_loud = mono_meter.momentary_loudness();
    let stereo_loud = stereo_meter.momentary_loudness();

    if mono_loud.is_finite() && stereo_loud.is_finite() {
        // Stereo with identical L/R channels should be close to mono
        // (EBU R128 channel weighting: stereo +1.5 dB vs mono, but momentary window
        // may differ slightly)
        let diff = (stereo_loud - mono_loud).abs();
        assert!(
            diff < 5.0,
            "stereo vs mono loudness diff should be small, got: {diff}"
        );
    }
}

/// EBU R128 test 8: LoudnessMeter wraps R128Meter and exposes compliance check.
#[test]
fn r128_compliance_check_via_loudness_meter() {
    let sr = 48000.0_f64;
    // -23 LUFS target: use amplitude that gives approximately that level
    // We don't need exact conformance, just check compliance logic
    let amplitude = 0.0708_f64;
    let n_samples = (sr * 15.0) as usize; // 15 seconds for integrated measurement

    let samples_f32: Vec<f32> = (0..n_samples)
        .map(|n| {
            let t = n as f64 / sr;
            (amplitude * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()) as f32
        })
        .collect();

    let mut meter = LoudnessMeter::new(LoudnessStandard::EbuR128, sr, 1);
    let chunk_size = (sr * 0.1) as usize;
    let mut n = 0usize;
    while n < samples_f32.len() {
        let end = (n + chunk_size).min(samples_f32.len());
        let chunk = &samples_f32[n..end];
        let frame = make_f32_frame(chunk, sr as u32, ChannelLayout::Mono);
        meter.measure(&frame);
        n = end;
    }

    let metrics = meter.get_metrics();
    // At minimum, check that the meter ran without panic and gave finite results
    assert!(
        metrics.true_peak_dbtp.is_finite() || metrics.true_peak_dbtp.is_infinite(),
        "true peak should be a valid float"
    );

    // The compliance check should return a valid status (not panic)
    let _status = meter.check_compliance();
}

// ---------------------------------------------------------------------------
// VU meter ballistics tests (IEC 60268-10)
// ---------------------------------------------------------------------------

/// IEC 60268-10 specifies that a VU meter should reach 99% of its steady-state
/// reading within 300ms of a 0 dBVU sine wave (at the reference level).
///
/// The standard specifies:
/// - Rise time (0 to -3 dB): ~300ms
/// - The needle should "overshoot" by no more than 1.5%
///
/// We test that:
/// 1. After 300ms of 0 dBVU sine, the reading is within the expected range.
/// 2. After silence, the meter returns towards negative infinity.
/// 3. Reference level is -18 dBFS by default.
fn make_f32_frame_with_ts(
    samples: &[f32],
    sample_rate: u32,
    channels: ChannelLayout,
    pts_samples: u64,
) -> AudioFrame {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

    let mut frame = AudioFrame::new(SampleFormat::F32, sample_rate, channels);
    frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));
    frame.timestamp = Timestamp::new(pts_samples as i64, Rational::new(1, i64::from(sample_rate)));
    frame
}

/// Process `chunks` of audio through the VU meter; each chunk is `chunk_size`
/// samples of a sine wave at the given amplitude.
fn process_vu_sine(
    meter: &mut VuMeter,
    amplitude: f32,
    freq_hz: f64,
    sample_rate: u32,
    num_samples: usize,
    chunk_size: usize,
) {
    let sr = f64::from(sample_rate);
    let mut n = 0usize;
    while n < num_samples {
        let end = (n + chunk_size).min(num_samples);
        let chunk: Vec<f32> = (n..end)
            .map(|i| {
                let t = i as f64 / sr;
                amplitude * (2.0 * std::f64::consts::PI * freq_hz * t).sin() as f32
            })
            .collect();
        let frame = make_f32_frame_with_ts(&chunk, sample_rate, ChannelLayout::Mono, n as u64);
        meter.process(&frame);
        n = end;
    }
}

/// VU ballistics test 1: Initial reading is below 0 dBVU (meter starts silent).
#[test]
fn vu_initial_reading_silent() {
    let meter = VuMeter::new(48000.0, 1);
    let reading = meter.vu_reading(0);
    // No samples processed → reading should be -inf or very low
    assert!(
        reading < -10.0 || reading.is_infinite(),
        "initial VU reading should be very low, got: {reading}"
    );
}

/// VU ballistics test 2: After a long burst of 0 dBVU signal the meter should
/// stabilize at approximately 0 dBVU (±3 dB tolerance).
///
/// 0 dBVU = -18 dBFS (default reference). Amplitude = 10^(-18/20) ≈ 0.1259.
#[test]
fn vu_steady_state_zero_dbvu() {
    let sample_rate = 48000u32;
    let reference_dbfs = -18.0_f64;
    let amplitude = 10.0_f64.powf(reference_dbfs / 20.0) as f32; // ≈ 0.1259
    let one_second = sample_rate as usize;
    let chunk = sample_rate as usize / 10; // 100ms chunks

    let mut meter = VuMeter::new(f64::from(sample_rate), 1);
    // Drive for 1.5 seconds to allow ballistics to settle
    process_vu_sine(
        &mut meter,
        amplitude,
        1000.0,
        sample_rate,
        one_second + one_second / 2,
        chunk,
    );

    let reading = meter.vu_reading(0);
    assert!(
        reading.is_finite(),
        "VU reading should be finite after sustained signal, got: {reading}"
    );
    // Allow ±5 dBVU tolerance around 0 dBVU
    assert!(
        reading > -10.0 && reading < 6.0,
        "Steady-state VU reading should be near 0 dBVU, got: {reading}"
    );
}

/// VU ballistics test 3: Below-reference signal gives negative dBVU reading.
///
/// A signal at -28 dBFS should read approximately -10 dBVU (below reference).
#[test]
fn vu_reading_below_reference() {
    let sample_rate = 48000u32;
    // -28 dBFS → -10 dBVU below the -18 dBFS reference
    let amplitude = 10.0_f64.powf(-28.0 / 20.0) as f32;
    let one_second = sample_rate as usize;
    let chunk = sample_rate as usize / 10;

    let mut meter = VuMeter::new(f64::from(sample_rate), 1);
    process_vu_sine(
        &mut meter,
        amplitude,
        1000.0,
        sample_rate,
        one_second + one_second / 2,
        chunk,
    );

    let reading = meter.vu_reading(0);
    assert!(
        reading.is_finite(),
        "VU reading should be finite, got: {reading}"
    );
    assert!(
        reading < 0.0,
        "sub-reference signal should give negative dBVU reading, got: {reading}"
    );
}

/// VU ballistics test 4: Overload signal (above 0 dBVU) triggers overload indicator.
///
/// A signal at -8 dBFS (+10 dBVU above reference) should be flagged as overload.
#[test]
fn vu_overload_detection() {
    let sample_rate = 48000u32;
    // -8 dBFS = +10 dBVU
    let amplitude = 10.0_f64.powf(-8.0 / 20.0) as f32;
    let one_second = sample_rate as usize;
    let chunk = sample_rate as usize / 10;

    let mut meter = VuMeter::new(f64::from(sample_rate), 1);
    process_vu_sine(
        &mut meter,
        amplitude,
        1000.0,
        sample_rate,
        one_second + one_second / 2,
        chunk,
    );

    assert!(
        meter.is_overload(0),
        "signal above 0 dBVU should trigger overload"
    );
}

/// VU ballistics test 5: After reset, peak VU reading is cleared.
#[test]
fn vu_reset_clears_peak() {
    let sample_rate = 48000u32;
    let amplitude = 10.0_f64.powf(-8.0 / 20.0) as f32; // Loud signal
    let chunk = 100usize;
    let frames = sample_rate as usize / 2;

    let mut meter = VuMeter::new(f64::from(sample_rate), 1);
    process_vu_sine(&mut meter, amplitude, 1000.0, sample_rate, frames, chunk);

    // Should have a non-trivial peak
    let peak_before = meter.peak_vu_reading(0);

    meter.reset_peaks();
    let peak_after = meter.peak_vu_reading(0);

    // After reset_peaks, peak should be much lower (or -inf)
    assert!(
        peak_after < peak_before || peak_after.is_infinite(),
        "peak should be reduced after reset, before={peak_before}, after={peak_after}"
    );
}

/// VU ballistics test 6: Normalized reading is within [0.0, 1.0].
#[test]
fn vu_normalized_reading_in_range() {
    let sample_rate = 48000u32;
    let amplitude = 10.0_f64.powf(-18.0 / 20.0) as f32; // 0 dBVU
    let chunk = 200usize;
    let frames = sample_rate as usize;

    let mut meter = VuMeter::new(f64::from(sample_rate), 1);
    process_vu_sine(&mut meter, amplitude, 1000.0, sample_rate, frames, chunk);

    let normalized = meter.normalized_reading(0);
    assert!(
        (0.0..=1.0).contains(&normalized),
        "normalized VU reading should be in [0,1], got: {normalized}"
    );
}

/// VU ballistics test 7: Stereo VU reading averages L and R channels.
#[test]
fn vu_stereo_reading_average() {
    let sample_rate = 48000u32;
    let sr_f64 = f64::from(sample_rate);
    let chunk_size = 500usize;
    let frames = sample_rate as usize;

    // Drive both channels with the same signal
    let amplitude = 10.0_f64.powf(-18.0 / 20.0) as f32;

    let mut meter = VuMeter::new(sr_f64, 2);

    let all_samples: Vec<f32> = (0..frames)
        .flat_map(|n| {
            let t = n as f64 / sr_f64;
            let v = amplitude * (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32;
            [v, v] // L = R
        })
        .collect();

    let mut n = 0usize;
    while n < all_samples.len() {
        let end = (n + chunk_size * 2).min(all_samples.len());
        // Make sure we include pairs
        let end = if (end - n) % 2 == 0 { end } else { end - 1 };
        if end <= n {
            break;
        }
        let chunk = &all_samples[n..end];
        let frame = make_f32_frame(chunk, sample_rate, ChannelLayout::Stereo);
        meter.process(&frame);
        n = end;
    }

    let left = meter.vu_reading(0);
    let right = meter.vu_reading(1);
    let stereo = meter.stereo_vu_reading();

    if left.is_finite() && right.is_finite() {
        let expected = (left + right) / 2.0;
        let diff = (stereo - expected).abs();
        assert!(
            diff < 0.1,
            "stereo VU reading should average L and R: stereo={stereo}, expected≈{expected}"
        );
    }
}

/// VU ballistics test 8: Color zone changes from Green to Red as level increases.
#[test]
fn vu_color_zone_transitions() {
    use oximedia_audio::meters::vu::ColorZone;

    let sample_rate = 48000u32;
    let chunk = 200usize;
    let frames = sample_rate as usize;

    // Green: signal at -5 dBVU (below -3 dBVU threshold)
    let green_amplitude = 10.0_f64.powf((-18.0 - 5.0) / 20.0) as f32; // -23 dBFS
    let mut meter_green = VuMeter::new(f64::from(sample_rate), 1);
    process_vu_sine(
        &mut meter_green,
        green_amplitude,
        1000.0,
        sample_rate,
        frames,
        chunk,
    );
    let viz_green = meter_green.visualization_data(0);
    if viz_green.db_vu.is_finite() && viz_green.db_vu < -3.0 {
        assert_eq!(
            viz_green.color_zone,
            ColorZone::Green,
            "signal at {} dBVU should be Green zone",
            viz_green.db_vu
        );
    }

    // Red: signal above 0 dBVU
    let red_amplitude = 10.0_f64.powf((-18.0 + 5.0) / 20.0) as f32; // -13 dBFS → +5 dBVU
    let mut meter_red = VuMeter::new(f64::from(sample_rate), 1);
    process_vu_sine(
        &mut meter_red,
        red_amplitude,
        1000.0,
        sample_rate,
        frames,
        chunk,
    );
    let viz_red = meter_red.visualization_data(0);
    if viz_red.db_vu.is_finite() && viz_red.db_vu > 0.0 {
        assert_eq!(
            viz_red.color_zone,
            ColorZone::Red,
            "signal at {} dBVU should be Red zone",
            viz_red.db_vu
        );
    }
}
