# oximedia-audio TODO

## Current Status
- 100+ source files across codecs (Opus, Vorbis, FLAC, MP3, PCM), DSP (biquad, compressor, delay, EQ, limiter, reverb), effects (chorus, flanger, phaser, LFO), spectrum (FFT, analyzer, spectrogram, waveform, features), fingerprint (constellation, hash, matching, database), loudness (EBU R128, ATSC A/85, K-weighting, gating, true peak), meters (VU, PPM, peak, RMS, correlation, goniometer, Dolby, ITU), spatial (ambisonics, binaural, panning, reverb), description (ducking, mixing, synthesis, timing, metadata)
- `AudioDecoder`/`AudioEncoder` traits, `AudioFrame`, `ChannelLayout`, `Resampler`
- Feature-gated codecs: opus, vorbis, flac, mp3, pcm
- Dependencies: oximedia-core, rubato, audioadapter, oxifft, bytes

## Enhancements
- [x] Add gapless playback support with proper encoder delay/padding handling in codec traits
- [x] Implement true peak limiter in `loudness/peak` with 4x oversampled detection
- [x] Add multi-band compressor in `compressor` (crossover network + per-band compression)
- [x] Implement look-ahead delay in `compressor` and `gate` for attack anticipation
- [x] Add wet/dry mix parameter to all `effects` (chorus, flanger, phaser)
- [x] Implement sidechain input for `compressor` and `gate` (external key signal)
- [ ] Add auto-gain in `loudness/normalize` to maintain consistent output level after processing
- [x] Implement `Resampler` quality presets (draft/good/best) mapping to rubato configurations
- [x] Add `AudioFrame` format conversion utilities (interleaved <-> planar, bit depth conversion)
- [ ] Implement FLAC encoder compression level parameter (0-8) in `flac/encoder`

## New Features
- [ ] Add AAC decoder (patent-free since 2023) as feature-gated module
- [ ] Implement ALAC (Apple Lossless) decoder for Apple ecosystem compatibility
- [ ] Add WAV file reader/writer with full RIFF chunk handling
- [ ] Implement audio watermarking module (embed/detect inaudible watermarks)
- [ ] Add noise reduction module (spectral subtraction, Wiener filter)
- [ ] Implement click/pop removal for vinyl restoration workflows
- [ ] Add convolution reverb using impulse response loading
- [ ] Implement graphic equalizer (31-band ISO standard) using `biquad` banks
- [ ] Add audio ducking module (auto-duck music under voiceover)
- [ ] Implement Dolby Atmos object metadata parsing for spatial audio rendering

## Performance
- [x] Replace `rustfft` with OxiFFT per COOLJAPAN Policy
- [ ] Add SIMD-optimized sample format conversion in `format_convert`
- [ ] Implement lock-free ring buffer for real-time audio threading in `stream_buffer`
- [ ] Optimize `biquad` filter with direct form II transposed for better numerical behavior
- [ ] Add batch processing mode to `meters` (process multiple channels simultaneously)
- [ ] Implement FFT plan caching in `spectrum/fft` to avoid repeated planner allocation
- [ ] Optimize Vorbis MDCT with split-radix algorithm in `vorbis/mdct`

## Testing
- [ ] Add FLAC round-trip test: encode -> decode -> bit-exact comparison
- [ ] Test Opus encoder/decoder with ITU-T P.862 PESQ-like quality metric
- [ ] Add `loudness` EBU R128 conformance test with EBU test signals
- [ ] Test `meters/vu` ballistics against IEC 60268-10 specified rise/fall times
- [ ] Test `spatial/ambisonics` encoding/decoding round-trip for 1st order
- [ ] Add `fingerprint` matching accuracy test with time-stretched and pitch-shifted audio
- [ ] Test `effects/chorus` with known LFO parameters and verify modulation depth

## Documentation
- [ ] Document codec feature gates and their compile-time implications
- [ ] Add DSP signal flow diagrams for compressor, reverb, and EQ chains
- [ ] Document `AudioFrame` memory layout and channel ordering conventions
