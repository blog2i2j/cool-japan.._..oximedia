//! Matrix mixer — flexible many-to-many channel routing with per-crosspoint gain.
//!
//! A matrix mixer allows any input bus to be routed to any output bus at an
//! independent gain level.  This is commonly used for monitor matrices,
//! broadcast routing, and studio patch-bay simulation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ────────────────────────────────────────────────────────────────────────────
// MatrixPoint
// ────────────────────────────────────────────────────────────────────────────

/// A single crosspoint in the routing matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixPoint {
    /// Input index (0-based).
    pub input: usize,
    /// Output index (0-based).
    pub output: usize,
    /// Gain at this crosspoint (0.0 = off, 1.0 = unity).
    pub gain: f32,
    /// Whether this crosspoint is active.
    pub enabled: bool,
}

impl MatrixPoint {
    /// Create an enabled crosspoint at unity gain.
    #[must_use]
    pub fn unity(input: usize, output: usize) -> Self {
        Self {
            input,
            output,
            gain: 1.0,
            enabled: true,
        }
    }

    /// Create a disabled (silent) crosspoint.
    #[must_use]
    pub fn silent(input: usize, output: usize) -> Self {
        Self {
            input,
            output,
            gain: 0.0,
            enabled: false,
        }
    }

    /// Effective gain: `gain` if enabled, otherwise `0.0`.
    #[must_use]
    pub fn effective_gain(&self) -> f32 {
        if self.enabled {
            self.gain
        } else {
            0.0
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MatrixMixer
// ────────────────────────────────────────────────────────────────────────────

/// A matrix mixer with `inputs × outputs` crosspoints.
#[derive(Debug, Clone)]
pub struct MatrixMixer {
    /// Number of input buses.
    pub inputs: usize,
    /// Number of output buses.
    pub outputs: usize,
    /// Flat array of crosspoints, indexed as `[input * outputs + output]`.
    points: Vec<MatrixPoint>,
    /// Optional label for each input.
    pub input_labels: Vec<String>,
    /// Optional label for each output.
    pub output_labels: Vec<String>,
}

impl MatrixMixer {
    /// Create a new matrix mixer with all crosspoints disabled (silent).
    #[must_use]
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let mut points = Vec::with_capacity(inputs * outputs);
        for i in 0..inputs {
            for o in 0..outputs {
                points.push(MatrixPoint::silent(i, o));
            }
        }
        Self {
            inputs,
            outputs,
            points,
            input_labels: (0..inputs).map(|i| format!("In {i}")).collect(),
            output_labels: (0..outputs).map(|o| format!("Out {o}")).collect(),
        }
    }

    /// Create an identity matrix (each input routed to its matching output at
    /// unity gain). Requires `inputs == outputs`.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let mut m = Self::new(size, size);
        for i in 0..size {
            m.set_gain(i, i, 1.0);
            m.set_enabled(i, i, true);
        }
        m
    }

    fn idx(&self, input: usize, output: usize) -> Option<usize> {
        if input < self.inputs && output < self.outputs {
            Some(input * self.outputs + output)
        } else {
            None
        }
    }

    /// Set the gain at crosspoint `(input, output)`, clamped to `[0.0, 2.0]`.
    pub fn set_gain(&mut self, input: usize, output: usize, gain: f32) {
        if let Some(idx) = self.idx(input, output) {
            self.points[idx].gain = gain.clamp(0.0, 2.0);
        }
    }

    /// Get the gain at crosspoint `(input, output)`.
    #[must_use]
    pub fn get_gain(&self, input: usize, output: usize) -> Option<f32> {
        self.idx(input, output).map(|idx| self.points[idx].gain)
    }

    /// Enable or disable crosspoint `(input, output)`.
    pub fn set_enabled(&mut self, input: usize, output: usize, enabled: bool) {
        if let Some(idx) = self.idx(input, output) {
            self.points[idx].enabled = enabled;
        }
    }

    /// Returns `true` if the crosspoint is enabled.
    #[must_use]
    pub fn is_enabled(&self, input: usize, output: usize) -> bool {
        self.idx(input, output)
            .is_some_and(|idx| self.points[idx].enabled)
    }

    /// Mix `input_levels` (one f32 per input) into an output buffer.
    ///
    /// Writes one summed value per output channel.
    #[must_use]
    pub fn mix(&self, input_levels: &[f32]) -> Vec<f32> {
        let n_in = input_levels.len().min(self.inputs);
        let mut output = vec![0.0f32; self.outputs];
        for i in 0..n_in {
            for o in 0..self.outputs {
                if let Some(idx) = self.idx(i, o) {
                    output[o] += input_levels[i] * self.points[idx].effective_gain();
                }
            }
        }
        output
    }

    /// Mute all crosspoints for a given output.
    pub fn mute_output(&mut self, output: usize) {
        for i in 0..self.inputs {
            self.set_enabled(i, output, false);
        }
    }

    /// Restore unity gain for all crosspoints on the diagonal (identity).
    pub fn reset_to_identity(&mut self) {
        for i in 0..self.inputs {
            for o in 0..self.outputs {
                let is_diagonal = i == o;
                self.set_enabled(i, o, is_diagonal);
                self.set_gain(i, o, if is_diagonal { 1.0 } else { 0.0 });
            }
        }
    }

    /// Return the number of active (enabled) crosspoints.
    #[must_use]
    pub fn active_crosspoint_count(&self) -> usize {
        self.points.iter().filter(|p| p.enabled).count()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_matrix_all_silent() {
        let m = MatrixMixer::new(4, 4);
        for i in 0..4 {
            for o in 0..4 {
                assert!(!m.is_enabled(i, o));
                assert!(
                    (m.get_gain(i, o).expect("get_gain should succeed") - 0.0).abs() < f32::EPSILON
                );
            }
        }
    }

    #[test]
    fn test_identity_matrix_diagonal_active() {
        let m = MatrixMixer::identity(3);
        assert!(m.is_enabled(0, 0));
        assert!(m.is_enabled(1, 1));
        assert!(m.is_enabled(2, 2));
        assert!(!m.is_enabled(0, 1));
        assert!(!m.is_enabled(1, 0));
    }

    #[test]
    fn test_set_and_get_gain() {
        let mut m = MatrixMixer::new(2, 2);
        m.set_gain(0, 1, 0.5);
        assert!((m.get_gain(0, 1).expect("get_gain should succeed") - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gain_clamped_above_max() {
        let mut m = MatrixMixer::new(2, 2);
        m.set_gain(0, 0, 5.0);
        assert!((m.get_gain(0, 0).expect("get_gain should succeed") - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gain_clamped_below_min() {
        let mut m = MatrixMixer::new(2, 2);
        m.set_gain(0, 0, -1.0);
        assert!((m.get_gain(0, 0).expect("get_gain should succeed") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_out_of_bounds_returns_none() {
        let m = MatrixMixer::new(2, 2);
        assert!(m.get_gain(5, 5).is_none());
        assert!(!m.is_enabled(5, 5));
    }

    #[test]
    fn test_mix_identity() {
        let m = MatrixMixer::identity(3);
        let inputs = vec![0.5, 0.7, 0.3];
        let outputs = m.mix(&inputs);
        assert!((outputs[0] - 0.5).abs() < 1e-6);
        assert!((outputs[1] - 0.7).abs() < 1e-6);
        assert!((outputs[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_mix_all_silent() {
        let m = MatrixMixer::new(2, 2);
        let inputs = vec![1.0, 1.0];
        let outputs = m.mix(&inputs);
        assert!((outputs[0]).abs() < f32::EPSILON);
        assert!((outputs[1]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mute_output() {
        let mut m = MatrixMixer::identity(3);
        m.mute_output(1);
        let inputs = vec![1.0, 1.0, 1.0];
        let outputs = m.mix(&inputs);
        // Output 1 should be 0 after muting.
        assert!((outputs[1]).abs() < f32::EPSILON);
        // Others unaffected.
        assert!((outputs[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_active_crosspoint_count_identity() {
        let m = MatrixMixer::identity(4);
        assert_eq!(m.active_crosspoint_count(), 4);
    }

    #[test]
    fn test_reset_to_identity() {
        let mut m = MatrixMixer::new(3, 3);
        m.set_enabled(0, 2, true);
        m.reset_to_identity();
        assert!(m.is_enabled(0, 0));
        assert!(!m.is_enabled(0, 2));
        assert_eq!(m.active_crosspoint_count(), 3);
    }

    #[test]
    fn test_matrix_point_effective_gain_disabled() {
        let p = MatrixPoint {
            input: 0,
            output: 0,
            gain: 1.0,
            enabled: false,
        };
        assert!((p.effective_gain() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_matrix_point_effective_gain_enabled() {
        let p = MatrixPoint::unity(0, 0);
        assert!((p.effective_gain() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_labels_generated_automatically() {
        let m = MatrixMixer::new(2, 3);
        assert_eq!(m.input_labels[0], "In 0");
        assert_eq!(m.output_labels[2], "Out 2");
    }
}
