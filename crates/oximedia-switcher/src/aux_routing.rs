// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Auxiliary audio routing matrix.
//!
//! `AuxMatrix` is an `inputs × outputs` crosspoint matrix for routing audio.
//! Any input can be routed to any output; multiple inputs routed to the same
//! output are summed (mixed).  The matrix state is stored as a boolean grid.

/// An auxiliary routing matrix for audio (or generic signal) routing.
///
/// The matrix models `inputs` source signals and `outputs` destination buses.
/// Each crosspoint `(input, output)` is either *enabled* (the input is mixed
/// into that output) or *disabled*.
pub struct AuxMatrix {
    /// Number of input channels.
    inputs: usize,
    /// Number of output buses.
    outputs: usize,
    /// `inputs × outputs` boolean grid; `matrix[i * outputs + o]` is `true`
    /// when input `i` is routed to output `o`.
    matrix: Vec<bool>,
}

impl AuxMatrix {
    /// Create a new `AuxMatrix` with all crosspoints disabled.
    ///
    /// # Arguments
    ///
    /// * `inputs`  – Number of input signal sources (> 0).
    /// * `outputs` – Number of output buses (> 0).
    ///
    /// Both are clamped to a minimum of 1.
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let inputs = inputs.max(1);
        let outputs = outputs.max(1);
        AuxMatrix {
            inputs,
            outputs,
            matrix: vec![false; inputs * outputs],
        }
    }

    /// Enable or disable the crosspoint at `(input, output)`.
    ///
    /// Out-of-range indices are silently ignored.
    pub fn route(&mut self, input: usize, output: usize, enabled: bool) {
        if input < self.inputs && output < self.outputs {
            self.matrix[input * self.outputs + output] = enabled;
        }
    }

    /// Return `true` if input `i` is routed to output `o`.
    pub fn is_routed(&self, input: usize, output: usize) -> bool {
        if input >= self.inputs || output >= self.outputs {
            return false;
        }
        self.matrix[input * self.outputs + output]
    }

    /// Clear all crosspoints (disable all routes).
    pub fn clear_all(&mut self) {
        for v in &mut self.matrix {
            *v = false;
        }
    }

    /// Enable exactly one input per output, clearing all others in that column.
    ///
    /// This models a traditional "exclusive select" switcher row.
    pub fn select_exclusive(&mut self, input: usize, output: usize) {
        if input >= self.inputs || output >= self.outputs {
            return;
        }
        for i in 0..self.inputs {
            self.matrix[i * self.outputs + output] = i == input;
        }
    }

    /// Mix the input sample buffers according to the current routing and return
    /// per-output mixed signals.
    ///
    /// # Arguments
    ///
    /// * `input_samples` – Slice of length `inputs`.  Each element is a `Vec<f32>`
    ///   of audio samples for that input.  If the slice is shorter than `inputs`,
    ///   missing inputs are treated as silence.
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<f32>>` of length `outputs`.  Each output buffer's length equals
    /// the maximum sample count among all inputs routed to it (zero-padded
    /// shorter buffers).  Outputs with no active inputs return an empty `Vec`.
    ///
    /// Output samples are clamped to `[-1.0, 1.0]`.
    pub fn mix(&self, input_samples: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut result: Vec<Vec<f32>> = (0..self.outputs).map(|_| Vec::new()).collect();

        for o in 0..self.outputs {
            // Determine max length for this output
            let max_len: usize = (0..self.inputs)
                .filter(|&i| self.is_routed(i, o))
                .map(|i| input_samples.get(i).map(|b| b.len()).unwrap_or(0))
                .max()
                .unwrap_or(0);

            if max_len == 0 {
                continue;
            }

            let mut mix = vec![0.0f32; max_len];
            for i in 0..self.inputs {
                if !self.is_routed(i, o) {
                    continue;
                }
                if let Some(buf) = input_samples.get(i) {
                    for (idx, &s) in buf.iter().enumerate() {
                        if idx < max_len {
                            mix[idx] += s;
                        }
                    }
                }
            }

            for s in &mut mix {
                *s = s.clamp(-1.0, 1.0);
            }
            result[o] = mix;
        }

        result
    }

    /// Number of inputs.
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// Number of outputs.
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// Return a list of all active (input, output) crosspoints.
    pub fn active_routes(&self) -> Vec<(usize, usize)> {
        let mut routes = Vec::new();
        for i in 0..self.inputs {
            for o in 0..self.outputs {
                if self.matrix[i * self.outputs + o] {
                    routes.push((i, o));
                }
            }
        }
        routes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_matrix_is_all_disabled() {
        let m = AuxMatrix::new(4, 4);
        for i in 0..4 {
            for o in 0..4 {
                assert!(!m.is_routed(i, o));
            }
        }
    }

    #[test]
    fn route_enables_crosspoint() {
        let mut m = AuxMatrix::new(4, 4);
        m.route(1, 2, true);
        assert!(m.is_routed(1, 2));
        assert!(!m.is_routed(0, 2));
    }

    #[test]
    fn route_disable_crosspoint() {
        let mut m = AuxMatrix::new(4, 4);
        m.route(1, 2, true);
        m.route(1, 2, false);
        assert!(!m.is_routed(1, 2));
    }

    #[test]
    fn out_of_bounds_ignored() {
        let mut m = AuxMatrix::new(2, 2);
        m.route(5, 5, true); // silent ignore
        assert!(!m.is_routed(5, 5));
    }

    #[test]
    fn mix_single_input_to_single_output() {
        let mut m = AuxMatrix::new(2, 2);
        m.route(0, 0, true);
        let input = vec![vec![0.5f32; 4], vec![0.2f32; 4]];
        let out = m.mix(&input);
        assert_eq!(out[0].len(), 4);
        for &s in &out[0] {
            assert!((s - 0.5).abs() < 1e-6);
        }
        assert!(out[1].is_empty()); // output 1 has no routes
    }

    #[test]
    fn mix_two_inputs_summed() {
        let mut m = AuxMatrix::new(2, 1);
        m.route(0, 0, true);
        m.route(1, 0, true);
        let input = vec![vec![0.3f32; 4], vec![0.2f32; 4]];
        let out = m.mix(&input);
        for &s in &out[0] {
            assert!((s - 0.5).abs() < 1e-5, "s={s}");
        }
    }

    #[test]
    fn mix_clamps_output() {
        let mut m = AuxMatrix::new(2, 1);
        m.route(0, 0, true);
        m.route(1, 0, true);
        let input = vec![vec![0.9f32; 2], vec![0.9f32; 2]];
        let out = m.mix(&input);
        for &s in &out[0] {
            assert!(s <= 1.0, "not clamped: {s}");
        }
    }

    #[test]
    fn select_exclusive_clears_column() {
        let mut m = AuxMatrix::new(3, 1);
        m.route(0, 0, true);
        m.route(1, 0, true);
        m.select_exclusive(2, 0);
        assert!(!m.is_routed(0, 0));
        assert!(!m.is_routed(1, 0));
        assert!(m.is_routed(2, 0));
    }

    #[test]
    fn active_routes_lists_enabled() {
        let mut m = AuxMatrix::new(3, 2);
        m.route(0, 0, true);
        m.route(2, 1, true);
        let routes = m.active_routes();
        assert_eq!(routes.len(), 2);
        assert!(routes.contains(&(0, 0)));
        assert!(routes.contains(&(2, 1)));
    }

    #[test]
    fn clear_all_disables_everything() {
        let mut m = AuxMatrix::new(4, 4);
        for i in 0..4 {
            m.route(i, i, true);
        }
        m.clear_all();
        assert!(m.active_routes().is_empty());
    }
}
