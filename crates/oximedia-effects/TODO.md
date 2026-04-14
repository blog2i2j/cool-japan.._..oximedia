# oximedia-effects TODO

## Current Status
- 73 source files spanning audio effects (reverb, delay, modulation, distortion, dynamics, filter, pitch, vocoder) and video effects (blend, chroma key, color grade, grain, lens flare, motion blur, vignette)
- Core `AudioEffect` trait with mono/stereo processing, reset, latency reporting
- Audio: Freeverb, plate reverb, convolution reverb, Schroeder reverb, room reverb, hall reverb
- Audio: Delay, multi-tap, ping-pong, tape echo; chorus, flanger, phaser, tremolo, vibrato, ring mod
- Audio: Overdrive, fuzz, bit crusher, waveshaper; gate, expander, compressor, de-esser, ducking
- Audio: Biquad, state variable, Moog ladder; pitch shifter, time stretch, harmonizer, auto-tune, vocoder
- Audio: Stereo widener, spatial audio, auto-pan, transient shaper, saturation
- Video: Blend modes, chroma key, luma key, barrel lens, chromatic aberration, composite, warp, glitch
- Dependencies: oximedia-core, oximedia-audio, oxifft, rubato, scirs2-core

## Enhancements
- [x] Replace `rustfft` with OxiFFT per COOLJAPAN policy in convolution reverb and pitch/vocoder
- [x] Add parameter smoothing to all effects to prevent zipper noise on real-time parameter changes
- [x] Implement true stereo processing in `reverb/freeverb.rs` with decorrelated L/R (prime offsets, per-channel diffusion)
- [x] Add sidechain input support to `compressor/mod.rs` (with SidechainFilter HPF/LPF/BPF) and `ducking.rs`
- [x] Enhance `pitch/autotune.rs` with chromatic scale, key-aware YIN pitch correction, 12-TET quantization
- [ ] Add wet/dry mix control to all effects via the `AudioEffect` trait
- [ ] Implement `eq/mod.rs` with parametric EQ bands (low shelf, high shelf, peaking, notch)
- [ ] Add feedback saturation modeling to `delay/delay.rs` for analog delay emulation
- [ ] Enhance `vocoder/channel.rs` with more analysis/synthesis filter bands (32+ bands)
- [x] Add oversampling option to `distortion/` effects to reduce aliasing artifacts

## New Features
- [ ] Implement convolution-based cabinet simulator for guitar/bass processing
- [ ] Add multi-band compressor splitting signal into low/mid/high bands
- [x] Implement lookahead limiter for broadcast loudness compliance
- [ ] Add spring reverb simulation using waveguide physical modeling
- [ ] Implement stereo-to-surround upmixer (5.1/7.1 channel support)
- [x] Add LUFS loudness metering effect (EBU R128 / ITU-R BS.1770)
- [ ] Implement granular synthesis time-stretcher as alternative to rubato
- [ ] Add video effect: motion vector-based optical flow slow motion
- [ ] Implement video effect: AI-free super resolution using edge-directed interpolation

## Performance
- [ ] Add SIMD-optimized biquad filter processing in `filter/mod.rs`
- [ ] Implement block-based FFT processing in `pitch/shifter.rs` to reduce per-sample overhead
- [ ] Use pre-allocated ring buffers in all delay-based effects instead of `Vec<f32>`
- [ ] Add double-buffering in `reverb/convolution.rs` for overlap-add processing
- [ ] Optimize `modulation/chorus.rs` LFO computation with wavetable lookup
- [ ] Profile `video/motion_blur.rs` and add frame accumulation caching

## Testing
- [ ] Add frequency response tests for all filter types in `filter/` (verify cutoff, Q, gain)
- [ ] Test `reverb/` effects for energy conservation (output energy <= input energy * wet+dry)
- [ ] Add latency compensation verification tests for all effects reporting non-zero latency
- [ ] Test `pitch/shifter.rs` pitch accuracy with sine wave inputs at known frequencies
- [ ] Add aliasing measurement tests for `distortion/` effects with oversampling on/off
- [ ] Test `video/chromakey.rs` with known green-screen test images

## Documentation
- [ ] Document the AudioEffect trait lifecycle (create, set_sample_rate, process, reset)
- [ ] Add signal flow diagrams for complex effects (reverb, vocoder, compressor)
- [ ] Document the video effects compositing pipeline and blend mode formulas
