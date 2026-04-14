// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Mix-minus audio bus.
//!
//! A *mix-minus* bus is an audio send that contains the mix of **all sources
//! except one** — typically used for IFB (interruptible feedback) and
//! teleconferencing to prevent echo: each remote participant receives the
//! programme mix minus their own contribution.
//!
//! `MixMinusBus` accumulates per-source float PCM contribution buffers and can
//! produce a mix-minus output for any given source ID.

use std::collections::HashMap;

/// A mix-minus audio bus.
///
/// Sources are registered by a numeric ID; each source contributes a `Vec<f32>`
/// of samples.  All source buffers must have the same length for a coherent mix;
/// shorter buffers are zero-padded internally.
pub struct MixMinusBus {
    /// Per-source sample buffers, keyed by source ID.
    sources: HashMap<u32, Vec<f32>>,
    /// Output gain applied to the summed mix (linear, 1.0 = unity).
    output_gain: f32,
}

impl MixMinusBus {
    /// Create an empty mix-minus bus with unity output gain.
    pub fn new() -> Self {
        MixMinusBus {
            sources: HashMap::new(),
            output_gain: 1.0,
        }
    }

    /// Create a mix-minus bus with a custom output gain.
    pub fn with_gain(output_gain: f32) -> Self {
        MixMinusBus {
            sources: HashMap::new(),
            output_gain: output_gain.max(0.0),
        }
    }

    /// Add or replace the audio contribution for `id`.
    ///
    /// If a source with this `id` already exists its buffer is replaced.
    pub fn add_source(&mut self, id: u32, samples: Vec<f32>) {
        self.sources.insert(id, samples);
    }

    /// Remove source `id` from the bus.
    pub fn remove_source(&mut self, id: u32) {
        self.sources.remove(&id);
    }

    /// Return `true` if source `id` is registered.
    pub fn has_source(&self, id: u32) -> bool {
        self.sources.contains_key(&id)
    }

    /// Number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Return the mix of all sources **except** `exclude_id`.
    ///
    /// Output length equals the length of the longest source buffer.
    /// Samples are summed and then multiplied by `output_gain`.
    /// The result is clamped to `[-1.0, 1.0]`.
    ///
    /// Returns an empty `Vec` if no sources remain after exclusion.
    pub fn output_for(&self, exclude_id: u32) -> Vec<f32> {
        // Determine output length: longest buffer among included sources
        let max_len = self
            .sources
            .iter()
            .filter(|(&id, _)| id != exclude_id)
            .map(|(_, buf)| buf.len())
            .max()
            .unwrap_or(0);

        if max_len == 0 {
            return Vec::new();
        }

        let mut mix = vec![0.0f32; max_len];

        for (&id, buf) in &self.sources {
            if id == exclude_id {
                continue;
            }
            for (i, &s) in buf.iter().enumerate() {
                if i < max_len {
                    mix[i] += s;
                }
            }
        }

        // Apply output gain and clamp
        for s in &mut mix {
            *s = (*s * self.output_gain).clamp(-1.0, 1.0);
        }

        mix
    }

    /// Return the full mix including all sources.
    pub fn full_mix(&self) -> Vec<f32> {
        // Use a non-existent ID that no source will have (u32::MAX)
        // Since all sources are included we need to not exclude anything.
        let max_len = self.sources.values().map(|b| b.len()).max().unwrap_or(0);
        if max_len == 0 {
            return Vec::new();
        }

        let mut mix = vec![0.0f32; max_len];
        for buf in self.sources.values() {
            for (i, &s) in buf.iter().enumerate() {
                if i < max_len {
                    mix[i] += s;
                }
            }
        }
        for s in &mut mix {
            *s = (*s * self.output_gain).clamp(-1.0, 1.0);
        }
        mix
    }

    /// Clear all sources.
    pub fn clear(&mut self) {
        self.sources.clear();
    }

    /// Set the output gain.
    pub fn set_gain(&mut self, gain: f32) {
        self.output_gain = gain.max(0.0);
    }

    /// Get the output gain.
    pub fn gain(&self) -> f32 {
        self.output_gain
    }
}

impl Default for MixMinusBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_bus_is_empty() {
        let bus = MixMinusBus::new();
        assert_eq!(bus.source_count(), 0);
    }

    #[test]
    fn add_source_registered() {
        let mut bus = MixMinusBus::new();
        bus.add_source(1, vec![0.5; 100]);
        assert!(bus.has_source(1));
        assert_eq!(bus.source_count(), 1);
    }

    #[test]
    fn output_for_excludes_target() {
        let mut bus = MixMinusBus::new();
        bus.add_source(1, vec![0.4; 4]);
        bus.add_source(2, vec![0.3; 4]);
        bus.add_source(3, vec![0.2; 4]);

        // Mix-minus for source 2: should include 1 + 3 = 0.6 per sample
        let out = bus.output_for(2);
        for &s in &out {
            assert!((s - 0.6).abs() < 1e-5, "sample={s}");
        }
    }

    #[test]
    fn output_for_only_source_returns_empty() {
        let mut bus = MixMinusBus::new();
        bus.add_source(1, vec![0.5; 10]);
        let out = bus.output_for(1);
        assert!(out.is_empty());
    }

    #[test]
    fn output_for_unknown_id_returns_full_mix() {
        let mut bus = MixMinusBus::new();
        bus.add_source(1, vec![0.2; 4]);
        bus.add_source(2, vec![0.3; 4]);
        // Excluding id=99 (not present) → full mix
        let out = bus.output_for(99);
        for &s in &out {
            assert!((s - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn output_clamps_to_minus_one_plus_one() {
        let mut bus = MixMinusBus::new();
        bus.add_source(1, vec![1.0; 4]);
        bus.add_source(2, vec![1.0; 4]);
        let out = bus.output_for(99); // all sources included
        for &s in &out {
            assert!(s <= 1.0, "not clamped: {s}");
        }
    }

    #[test]
    fn full_mix_sums_all() {
        let mut bus = MixMinusBus::new();
        bus.add_source(1, vec![0.1; 4]);
        bus.add_source(2, vec![0.2; 4]);
        let mix = bus.full_mix();
        for &s in &mix {
            assert!((s - 0.3).abs() < 1e-5);
        }
    }

    #[test]
    fn remove_source() {
        let mut bus = MixMinusBus::new();
        bus.add_source(5, vec![0.5; 10]);
        bus.remove_source(5);
        assert!(!bus.has_source(5));
        assert_eq!(bus.source_count(), 0);
    }

    #[test]
    fn gain_applied() {
        let mut bus = MixMinusBus::with_gain(0.5);
        bus.add_source(1, vec![0.4; 4]);
        bus.add_source(2, vec![0.2; 4]);
        let out = bus.output_for(2); // only source 1: 0.4 * 0.5 = 0.2
        for &s in &out {
            assert!((s - 0.2).abs() < 1e-5, "s={s}");
        }
    }
}
