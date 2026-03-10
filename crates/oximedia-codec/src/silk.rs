//! Standalone SILK frame decoding types and scaffolding.
//!
//! SILK (Skype Low Latency Audio Codec) is the speech codec used within Opus.
//! This module provides lightweight frame-level types for parsing SILK frame
//! headers and applying LPC synthesis, independent of the full Opus decoder
//! pipeline.

/// SILK operating bandwidth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilkBandwidth {
    /// Narrowband — 8 kHz sample rate.
    NarrowBand,
    /// Medium band — 12 kHz sample rate.
    MediumBand,
    /// Wideband — 16 kHz sample rate.
    WideBand,
    /// Super-wideband — 24 kHz sample rate.
    SuperWideBand,
}

impl SilkBandwidth {
    /// Returns the sample rate in Hz associated with this bandwidth.
    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::NarrowBand => 8_000,
            Self::MediumBand => 12_000,
            Self::WideBand => 16_000,
            Self::SuperWideBand => 24_000,
        }
    }
}

/// Parsed header fields from a SILK frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SilkFrameHeader {
    /// Voice activity detection flag.
    pub vad_flag: bool,
    /// Low Bitrate Redundancy (LBRR) payload present.
    pub lbrr_flag: bool,
    /// Signal type: 0 = inactive, 1 = voiced, 2 = unvoiced.
    pub signal_type: u8,
    /// Quantization offset type (0 or 1).
    pub quantization_offset: u8,
}

impl SilkFrameHeader {
    /// Parses a `SilkFrameHeader` from the first byte(s) of a raw frame.
    ///
    /// The SILK frame header is packed into the leading bits of the payload.
    /// This parser reads the minimum information needed to scaffold frame
    /// processing; full SILK decoding requires an entropy/range decoder.
    ///
    /// Layout of byte 0:
    /// ```text
    /// Bit 7: VAD flag
    /// Bit 6: LBRR flag
    /// Bits 5-4: signal_type (0-2, values 0b11 treated as inactive)
    /// Bit 3: quantization_offset
    /// Bits 2-0: reserved / additional payload bits
    /// ```
    pub fn parse(data: &[u8]) -> Result<Self, String> {
        if data.is_empty() {
            return Err("SILK frame data is empty".to_string());
        }

        let b0 = data[0];

        let vad_flag = (b0 & 0x80) != 0;
        let lbrr_flag = (b0 & 0x40) != 0;
        let raw_signal = (b0 >> 4) & 0x03;
        let signal_type = if raw_signal > 2 { 0 } else { raw_signal };
        let quantization_offset = (b0 >> 3) & 0x01;

        Ok(Self {
            vad_flag,
            lbrr_flag,
            signal_type,
            quantization_offset,
        })
    }
}

/// LPC (Linear Predictive Coding) filter coefficients for one SILK subframe.
///
/// Coefficients are stored in Q12 fixed-point format (i.e. the value `4096`
/// represents 1.0).
#[derive(Debug, Clone)]
pub struct SilkLpcCoeffs {
    /// LPC filter order (10 for narrowband/mediumband, 16 for wideband/super-wideband).
    pub order: usize,
    /// LPC filter coefficients in Q12 fixed-point.
    pub coeffs: Vec<i16>,
}

impl SilkLpcCoeffs {
    /// Creates a zeroed `SilkLpcCoeffs` with the given filter order.
    pub fn new(order: usize) -> Self {
        Self {
            order,
            coeffs: vec![0i16; order],
        }
    }
}

/// A decoded SILK frame.
#[derive(Debug, Clone)]
pub struct SilkFrame {
    /// Parsed frame header.
    pub header: SilkFrameHeader,
    /// LPC coefficients used for synthesis.
    pub lpc: SilkLpcCoeffs,
    /// Decoded PCM samples (i16, linear).
    pub samples: Vec<i16>,
    /// Number of samples in this frame.
    pub sample_count: usize,
}

impl SilkFrame {
    /// Creates an empty `SilkFrame` with default-zeroed fields.
    pub fn new() -> Self {
        Self {
            header: SilkFrameHeader {
                vad_flag: false,
                lbrr_flag: false,
                signal_type: 0,
                quantization_offset: 0,
            },
            lpc: SilkLpcCoeffs::new(16),
            samples: Vec::new(),
            sample_count: 0,
        }
    }

    /// Returns the number of PCM samples in this frame.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the samples normalised to the range `[-1.0, 1.0]` as `f32`.
    pub fn as_f32_samples(&self) -> Vec<f32> {
        self.samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect()
    }
}

/// SILK frame decoder scaffold.
///
/// This type parses the SILK frame header and exposes helpers for LPC
/// synthesis. Full entropy-coded SILK decoding is extremely complex and is
/// provided by the Opus implementation in `crate::opus::silk`. This struct
/// is intentionally lightweight and suitable for testing and scaffolding.
#[derive(Debug)]
pub struct SilkDecoder {
    /// Operating bandwidth.
    pub bandwidth: SilkBandwidth,
    /// Expected frame size in samples.
    pub frame_size: usize,
    /// Previous output samples kept for LPC synthesis state (history).
    pub prev_samples: Vec<i16>,
}

impl SilkDecoder {
    /// Creates a new `SilkDecoder` for the given bandwidth.
    pub fn new(bandwidth: SilkBandwidth) -> Self {
        // SILK uses 20 ms frames.
        let frame_size = (bandwidth.sample_rate() as usize) * 20 / 1000;
        // LPC order is 16 for WB/SWB, 10 for NB/MB.
        let lpc_order = match bandwidth {
            SilkBandwidth::NarrowBand | SilkBandwidth::MediumBand => 10,
            SilkBandwidth::WideBand | SilkBandwidth::SuperWideBand => 16,
        };
        Self {
            bandwidth,
            frame_size,
            prev_samples: vec![0i16; lpc_order],
        }
    }

    /// Parses the frame header and returns a `SilkFrame` with zeroed samples.
    ///
    /// Full SILK decoding (excitation decoding, LTP, noise shaping …) is not
    /// implemented here; the goal of this method is to validate the header and
    /// set up the frame scaffold so that higher-level code can fill in the
    /// decoded samples.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<SilkFrame, String> {
        let header = SilkFrameHeader::parse(data)?;

        let lpc_order = self.prev_samples.len();
        let lpc = SilkLpcCoeffs::new(lpc_order);

        let samples = vec![0i16; self.frame_size];
        let sample_count = self.frame_size;

        Ok(SilkFrame {
            header,
            lpc,
            samples,
            sample_count,
        })
    }

    /// Applies the LPC synthesis filter to an excitation signal.
    ///
    /// The synthesis filter is:
    /// ```text
    /// s[n] = excitation[n] + sum_{k=0}^{order-1} (lpc.coeffs[k] * s[n-k-1]) / 4096
    /// ```
    /// where the division by 4096 converts Q12 fixed-point coefficients back
    /// to the integer domain.
    ///
    /// The decoder's `prev_samples` buffer is used as the initial state and
    /// is updated to the last `order` samples of the output on return.
    pub fn apply_lpc_synthesis(&mut self, excitation: &[i16], lpc: &SilkLpcCoeffs) -> Vec<i16> {
        let order = lpc.order;
        let n = excitation.len();
        let mut output = vec![0i16; n];

        // Combined history: prev_samples (oldest…newest) ++ output produced so far.
        // We index as: index_in_history(-1) = prev_samples[order - 1], etc.
        let history: Vec<i16> = self.prev_samples.clone();

        for i in 0..n {
            let mut acc: i64 = excitation[i] as i64;
            for k in 0..order {
                // s[n - k - 1]: look back k+1 samples.
                let back = k + 1;
                let sample = if back <= i {
                    output[i - back] as i64
                } else {
                    // Still in the prev_samples history.
                    let hist_idx = order as isize - (back as isize - i as isize);
                    if hist_idx >= 0 && (hist_idx as usize) < history.len() {
                        history[hist_idx as usize] as i64
                    } else {
                        0i64
                    }
                };
                acc += (lpc.coeffs[k] as i64 * sample) >> 12;
            }
            output[i] = acc.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        }

        // Update history with the last `order` output samples.
        let keep = order.min(n);
        let src_start = n - keep;
        for (dst, src) in self.prev_samples[order - keep..]
            .iter_mut()
            .zip(output[src_start..].iter())
        {
            *dst = *src;
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silk_bandwidth_sample_rate_narrowband() {
        assert_eq!(SilkBandwidth::NarrowBand.sample_rate(), 8_000);
    }

    #[test]
    fn test_silk_bandwidth_sample_rate_mediumband() {
        assert_eq!(SilkBandwidth::MediumBand.sample_rate(), 12_000);
    }

    #[test]
    fn test_silk_bandwidth_sample_rate_wideband() {
        assert_eq!(SilkBandwidth::WideBand.sample_rate(), 16_000);
    }

    #[test]
    fn test_silk_bandwidth_sample_rate_superwideband() {
        assert_eq!(SilkBandwidth::SuperWideBand.sample_rate(), 24_000);
    }

    #[test]
    fn test_silk_frame_header_parse_basic() {
        // Byte 0: VAD=1, LBRR=0, signal=01 (voiced), q_offset=0
        // 1 0 01 0 000 = 0b10010000 = 0x90
        let data = [0x90u8];
        let hdr = SilkFrameHeader::parse(&data).expect("should succeed");
        assert!(hdr.vad_flag);
        assert!(!hdr.lbrr_flag);
        assert_eq!(hdr.signal_type, 1);
        assert_eq!(hdr.quantization_offset, 0);
    }

    #[test]
    fn test_silk_frame_header_parse_empty_returns_error() {
        let result = SilkFrameHeader::parse(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_silk_decoder_new() {
        let dec = SilkDecoder::new(SilkBandwidth::WideBand);
        assert_eq!(dec.bandwidth, SilkBandwidth::WideBand);
        // 16000 Hz * 20ms = 320 samples
        assert_eq!(dec.frame_size, 320);
    }

    #[test]
    fn test_silk_decoder_decode_frame() {
        let mut dec = SilkDecoder::new(SilkBandwidth::NarrowBand);
        let data = [0x00u8; 10];
        let frame = dec.decode_frame(&data).expect("should succeed");
        // 8000 Hz * 20ms = 160 samples
        assert_eq!(frame.sample_count(), 160);
        assert_eq!(frame.samples.len(), 160);
    }

    #[test]
    fn test_silk_lpc_synthesis_zero_excitation_gives_zero_output() {
        let mut dec = SilkDecoder::new(SilkBandwidth::NarrowBand);
        let excitation = vec![0i16; 160];
        let lpc = SilkLpcCoeffs::new(10);
        let output = dec.apply_lpc_synthesis(&excitation, &lpc);
        assert!(output.iter().all(|&s| s == 0));
    }

    #[test]
    fn test_silk_frame_as_f32_samples_i16_max() {
        let mut frame = SilkFrame::new();
        frame.samples = vec![i16::MAX];
        frame.sample_count = 1;
        let f32s = frame.as_f32_samples();
        assert!((f32s[0] - 1.0f32).abs() < 1e-4);
    }

    #[test]
    fn test_silk_frame_as_f32_samples_i16_min() {
        let mut frame = SilkFrame::new();
        frame.samples = vec![i16::MIN];
        frame.sample_count = 1;
        let f32s = frame.as_f32_samples();
        // i16::MIN as f32 / i16::MAX as f32 ≈ -1.00003…
        assert!(f32s[0] < -0.999);
    }

    #[test]
    fn test_silk_lpc_coeffs_new() {
        let lpc = SilkLpcCoeffs::new(10);
        assert_eq!(lpc.order, 10);
        assert_eq!(lpc.coeffs.len(), 10);
        assert!(lpc.coeffs.iter().all(|&c| c == 0));
    }
}
