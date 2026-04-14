//! Crosspoint routing matrix implementation for any-to-any audio routing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a crosspoint in the routing matrix
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CrosspointId {
    /// Input channel index
    pub input: usize,
    /// Output channel index
    pub output: usize,
}

impl CrosspointId {
    /// Create a new crosspoint identifier
    #[must_use]
    pub const fn new(input: usize, output: usize) -> Self {
        Self { input, output }
    }
}

/// State of a crosspoint connection
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum CrosspointState {
    /// Crosspoint is disconnected
    #[default]
    Disconnected,
    /// Crosspoint is connected with optional gain (in dB)
    Connected { gain_db: f32 },
}

/// Crosspoint routing matrix for full any-to-any routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrosspointMatrix {
    /// Number of input channels
    inputs: usize,
    /// Number of output channels
    outputs: usize,
    /// Crosspoint states
    crosspoints: HashMap<CrosspointId, CrosspointState>,
    /// Input labels
    input_labels: Vec<String>,
    /// Output labels
    output_labels: Vec<String>,
}

impl CrosspointMatrix {
    /// Create a new crosspoint matrix
    #[must_use]
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            inputs,
            outputs,
            crosspoints: HashMap::new(),
            input_labels: (0..inputs).map(|i| format!("Input {}", i + 1)).collect(),
            output_labels: (0..outputs).map(|i| format!("Output {}", i + 1)).collect(),
        }
    }

    /// Get the number of inputs
    #[must_use]
    pub const fn input_count(&self) -> usize {
        self.inputs
    }

    /// Get the number of outputs
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.outputs
    }

    /// Connect an input to an output with optional gain
    pub fn connect(
        &mut self,
        input: usize,
        output: usize,
        gain_db: Option<f32>,
    ) -> Result<(), MatrixError> {
        if input >= self.inputs {
            return Err(MatrixError::InvalidInput(input));
        }
        if output >= self.outputs {
            return Err(MatrixError::InvalidOutput(output));
        }

        let crosspoint = CrosspointId::new(input, output);
        self.crosspoints.insert(
            crosspoint,
            CrosspointState::Connected {
                gain_db: gain_db.unwrap_or(0.0),
            },
        );
        Ok(())
    }

    /// Disconnect an input from an output
    pub fn disconnect(&mut self, input: usize, output: usize) -> Result<(), MatrixError> {
        if input >= self.inputs {
            return Err(MatrixError::InvalidInput(input));
        }
        if output >= self.outputs {
            return Err(MatrixError::InvalidOutput(output));
        }

        let crosspoint = CrosspointId::new(input, output);
        self.crosspoints
            .insert(crosspoint, CrosspointState::Disconnected);
        Ok(())
    }

    /// Check if an input is connected to an output
    #[must_use]
    pub fn is_connected(&self, input: usize, output: usize) -> bool {
        let crosspoint = CrosspointId::new(input, output);
        matches!(
            self.crosspoints.get(&crosspoint),
            Some(CrosspointState::Connected { .. })
        )
    }

    /// Get the state of a crosspoint
    #[must_use]
    pub fn get_state(&self, input: usize, output: usize) -> CrosspointState {
        let crosspoint = CrosspointId::new(input, output);
        self.crosspoints
            .get(&crosspoint)
            .copied()
            .unwrap_or(CrosspointState::Disconnected)
    }

    /// Set gain for a crosspoint (must be connected)
    pub fn set_gain(
        &mut self,
        input: usize,
        output: usize,
        gain_db: f32,
    ) -> Result<(), MatrixError> {
        if input >= self.inputs {
            return Err(MatrixError::InvalidInput(input));
        }
        if output >= self.outputs {
            return Err(MatrixError::InvalidOutput(output));
        }

        let crosspoint = CrosspointId::new(input, output);
        if let Some(state) = self.crosspoints.get_mut(&crosspoint) {
            if let CrosspointState::Connected { gain_db: gain } = state {
                *gain = gain_db;
                return Ok(());
            }
        }
        Err(MatrixError::NotConnected(input, output))
    }

    /// Get all inputs connected to an output
    #[must_use]
    pub fn get_inputs_for_output(&self, output: usize) -> Vec<usize> {
        (0..self.inputs)
            .filter(|&input| self.is_connected(input, output))
            .collect()
    }

    /// Get all outputs connected to an input
    #[must_use]
    pub fn get_outputs_for_input(&self, input: usize) -> Vec<usize> {
        (0..self.outputs)
            .filter(|&output| self.is_connected(input, output))
            .collect()
    }

    /// Set label for an input
    pub fn set_input_label(&mut self, input: usize, label: String) -> Result<(), MatrixError> {
        if input >= self.inputs {
            return Err(MatrixError::InvalidInput(input));
        }
        self.input_labels[input] = label;
        Ok(())
    }

    /// Set label for an output
    pub fn set_output_label(&mut self, output: usize, label: String) -> Result<(), MatrixError> {
        if output >= self.outputs {
            return Err(MatrixError::InvalidOutput(output));
        }
        self.output_labels[output] = label;
        Ok(())
    }

    /// Get label for an input
    #[must_use]
    pub fn get_input_label(&self, input: usize) -> Option<&str> {
        self.input_labels.get(input).map(String::as_str)
    }

    /// Get label for an output
    #[must_use]
    pub fn get_output_label(&self, output: usize) -> Option<&str> {
        self.output_labels.get(output).map(String::as_str)
    }

    /// Create a new crosspoint matrix suitable for large (≥256×256) routing.
    ///
    /// Semantically identical to [`Self::new`]; the internal representation is
    /// always sparse (HashMap), so this constructor is an alias that makes
    /// intent explicit at the call site.
    #[must_use]
    pub fn new_sparse(inputs: usize, outputs: usize) -> Self {
        Self::new(inputs, outputs)
    }

    /// Compute the mix bus output for a given `output` channel.
    ///
    /// Sums `input_samples[input] * linear_gain` for every connected input,
    /// skipping zero entries (sparse iteration).  Returns the accumulated sum.
    ///
    /// `input_samples` must have at least `self.inputs` entries; out-of-range
    /// inputs are skipped silently.
    #[must_use]
    pub fn mix_bus(&self, output: usize, input_samples: &[f32]) -> f32 {
        if output >= self.outputs {
            return 0.0;
        }
        let mut sum = 0.0_f32;
        for (id, state) in &self.crosspoints {
            if id.output != output {
                continue;
            }
            if let CrosspointState::Connected { gain_db } = state {
                if id.input < input_samples.len() {
                    let gain_linear = 10.0_f32.powf(gain_db / 20.0);
                    sum += input_samples[id.input] * gain_linear;
                }
            }
        }
        sum
    }

    /// Compute all output channels at once using sparse iteration.
    ///
    /// Returns a `Vec<f32>` of length `self.outputs`.
    #[must_use]
    pub fn mix_bus_all(&self, input_samples: &[f32]) -> Vec<f32> {
        let mut outputs = vec![0.0_f32; self.outputs];
        for (id, state) in &self.crosspoints {
            if let CrosspointState::Connected { gain_db } = state {
                if id.input < input_samples.len() && id.output < self.outputs {
                    let gain_linear = 10.0_f32.powf(gain_db / 20.0);
                    outputs[id.output] += input_samples[id.input] * gain_linear;
                }
            }
        }
        outputs
    }

    /// Clear all connections
    pub fn clear_all(&mut self) {
        self.crosspoints.clear();
    }

    /// Get all active crosspoints
    #[must_use]
    pub fn get_active_crosspoints(&self) -> Vec<(CrosspointId, f32)> {
        self.crosspoints
            .iter()
            .filter_map(|(id, state)| {
                if let CrosspointState::Connected { gain_db } = state {
                    Some((*id, *gain_db))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Errors that can occur in matrix operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum MatrixError {
    /// Invalid input index
    #[error("Invalid input index: {0}")]
    InvalidInput(usize),
    /// Invalid output index
    #[error("Invalid output index: {0}")]
    InvalidOutput(usize),
    /// Crosspoint not connected
    #[error("Crosspoint not connected: input {0} to output {1}")]
    NotConnected(usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crosspoint_id() {
        let cp = CrosspointId::new(5, 10);
        assert_eq!(cp.input, 5);
        assert_eq!(cp.output, 10);
    }

    #[test]
    fn test_matrix_creation() {
        let matrix = CrosspointMatrix::new(16, 8);
        assert_eq!(matrix.input_count(), 16);
        assert_eq!(matrix.output_count(), 8);
    }

    #[test]
    fn test_connect_disconnect() {
        let mut matrix = CrosspointMatrix::new(4, 4);

        assert!(!matrix.is_connected(0, 0));
        matrix
            .connect(0, 0, Some(0.0))
            .expect("should succeed in test");
        assert!(matrix.is_connected(0, 0));

        matrix.disconnect(0, 0).expect("should succeed in test");
        assert!(!matrix.is_connected(0, 0));
    }

    #[test]
    fn test_gain_control() {
        let mut matrix = CrosspointMatrix::new(4, 4);

        matrix
            .connect(0, 0, Some(-6.0))
            .expect("should succeed in test");
        if let CrosspointState::Connected { gain_db } = matrix.get_state(0, 0) {
            assert!((gain_db - (-6.0)).abs() < f32::EPSILON);
        } else {
            panic!("Expected connected state");
        }

        matrix.set_gain(0, 0, -3.0).expect("should succeed in test");
        if let CrosspointState::Connected { gain_db } = matrix.get_state(0, 0) {
            assert!((gain_db - (-3.0)).abs() < f32::EPSILON);
        } else {
            panic!("Expected connected state");
        }
    }

    #[test]
    fn test_labels() {
        let mut matrix = CrosspointMatrix::new(2, 2);

        matrix
            .set_input_label(0, "Mic 1".to_string())
            .expect("should succeed in test");
        matrix
            .set_output_label(1, "Speaker R".to_string())
            .expect("should succeed in test");

        assert_eq!(matrix.get_input_label(0), Some("Mic 1"));
        assert_eq!(matrix.get_output_label(1), Some("Speaker R"));
    }

    #[test]
    fn test_routing_queries() {
        let mut matrix = CrosspointMatrix::new(4, 4);

        matrix.connect(0, 0, None).expect("should succeed in test");
        matrix.connect(0, 1, None).expect("should succeed in test");
        matrix.connect(1, 0, None).expect("should succeed in test");

        let outputs = matrix.get_outputs_for_input(0);
        assert_eq!(outputs.len(), 2);
        assert!(outputs.contains(&0));
        assert!(outputs.contains(&1));

        let inputs = matrix.get_inputs_for_output(0);
        assert_eq!(inputs.len(), 2);
        assert!(inputs.contains(&0));
        assert!(inputs.contains(&1));
    }

    #[test]
    fn test_invalid_indices() {
        let mut matrix = CrosspointMatrix::new(4, 4);

        assert!(matches!(
            matrix.connect(10, 0, None),
            Err(MatrixError::InvalidInput(10))
        ));
        assert!(matches!(
            matrix.connect(0, 10, None),
            Err(MatrixError::InvalidOutput(10))
        ));
    }

    // -----------------------------------------------------------------------
    // Task B: Sparse / 256×256 / mix_bus tests (10 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_sparse_same_as_new() {
        let dense = CrosspointMatrix::new(256, 256);
        let sparse = CrosspointMatrix::new_sparse(256, 256);
        assert_eq!(dense.input_count(), sparse.input_count());
        assert_eq!(dense.output_count(), sparse.output_count());
        assert_eq!(dense.get_active_crosspoints().len(), 0);
        assert_eq!(sparse.get_active_crosspoints().len(), 0);
    }

    #[test]
    fn test_mix_bus_zero_output() {
        let matrix = CrosspointMatrix::new(4, 4);
        let samples = [1.0_f32, 2.0, 3.0, 4.0];
        // No connections → output should be 0.
        let out = matrix.mix_bus(0, &samples);
        assert!(out.abs() < 1e-10, "expected 0, got {out}");
    }

    #[test]
    fn test_mix_bus_single_connection_at_0db() {
        let mut matrix = CrosspointMatrix::new(4, 4);
        // input 0 → output 0 at 0 dB (linear gain = 1.0).
        matrix.connect(0, 0, Some(0.0)).expect("ok");
        let samples = [0.5_f32, 0.0, 0.0, 0.0];
        let out = matrix.mix_bus(0, &samples);
        assert!((out - 0.5).abs() < 1e-5, "expected 0.5, got {out}");
    }

    #[test]
    fn test_mix_bus_sum_multiple_inputs() {
        let mut matrix = CrosspointMatrix::new(4, 4);
        // input 0 → output 0 at 0 dB; input 1 → output 0 at 0 dB.
        matrix.connect(0, 0, Some(0.0)).expect("ok");
        matrix.connect(1, 0, Some(0.0)).expect("ok");
        let samples = [0.3_f32, 0.4, 0.0, 0.0];
        let out = matrix.mix_bus(0, &samples);
        // 0.3 + 0.4 = 0.7
        assert!((out - 0.7).abs() < 1e-5, "expected 0.7, got {out}");
    }

    #[test]
    fn test_mix_bus_with_gain() {
        let mut matrix = CrosspointMatrix::new(4, 4);
        // -20 dB → linear = 0.1
        matrix.connect(0, 0, Some(-20.0)).expect("ok");
        let samples = [1.0_f32, 0.0, 0.0, 0.0];
        let out = matrix.mix_bus(0, &samples);
        assert!((out - 0.1).abs() < 0.01, "expected ~0.1, got {out}");
    }

    #[test]
    fn test_mix_bus_out_of_bounds_output() {
        let matrix = CrosspointMatrix::new(4, 4);
        let samples = [1.0_f32; 4];
        // Output 10 is out of bounds → returns 0.
        let out = matrix.mix_bus(10, &samples);
        assert!(out.abs() < 1e-10);
    }

    #[test]
    fn test_mix_bus_all_matches_per_output() {
        let mut matrix = CrosspointMatrix::new(4, 4);
        matrix.connect(0, 0, Some(0.0)).expect("ok");
        matrix.connect(1, 1, Some(0.0)).expect("ok");
        matrix.connect(2, 2, Some(0.0)).expect("ok");
        matrix.connect(3, 3, Some(0.0)).expect("ok");
        let samples = [0.1_f32, 0.2, 0.3, 0.4];
        let all = matrix.mix_bus_all(&samples);
        assert_eq!(all.len(), 4);
        for (o, &expected) in samples.iter().enumerate() {
            let per = matrix.mix_bus(o, &samples);
            assert!(
                (all[o] - per).abs() < 1e-7,
                "mismatch at output {o}: all={} vs per={}",
                all[o],
                per
            );
            assert!(
                (all[o] - expected).abs() < 1e-5,
                "output {o} expected {expected}, got {}",
                all[o]
            );
        }
    }

    #[test]
    fn test_256x256_with_5pct_fill() {
        let n = 256;
        let target_connections = (n * n) * 5 / 100; // 5%
        let mut matrix = CrosspointMatrix::new_sparse(n, n);

        // Connect every 20th input to every 20th output (approximately 5% fill).
        let mut count = 0;
        'outer: for i in (0..n).step_by(4) {
            for o in (0..n).step_by(5) {
                if matrix.connect(i, o, Some(0.0)).is_ok() {
                    count += 1;
                }
                if count >= target_connections {
                    break 'outer;
                }
            }
        }

        let active = matrix.get_active_crosspoints();
        assert!(
            active.len() >= 1,
            "expected at least 1 active crosspoint, got {}",
            active.len()
        );
        // Verify mix_bus computes without panic.
        let samples = vec![0.5_f32; n];
        let _ = matrix.mix_bus(0, &samples);
        let _ = matrix.mix_bus_all(&samples);
    }

    #[test]
    fn test_connect_disconnect_roundtrip_mix_bus() {
        let mut matrix = CrosspointMatrix::new(4, 4);
        matrix.connect(0, 0, Some(0.0)).expect("ok");
        assert!((matrix.mix_bus(0, &[1.0_f32, 0.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);

        matrix.disconnect(0, 0).expect("ok");
        assert!(matrix.mix_bus(0, &[1.0_f32, 0.0, 0.0, 0.0]).abs() < 1e-10);
    }

    #[test]
    fn test_256x256_mix_bus_diagonal() {
        let n = 256;
        let mut matrix = CrosspointMatrix::new_sparse(n, n);
        // Connect diagonal at 0 dB.
        for i in 0..n {
            matrix.connect(i, i, Some(0.0)).expect("ok");
        }
        // Each output should receive exactly one input.
        let samples: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        for o in 0..n {
            let out = matrix.mix_bus(o, &samples);
            let expected = samples[o]; // diagonal: input o → output o
            assert!(
                (out - expected).abs() < 1e-5,
                "output {o}: expected {expected}, got {out}"
            );
        }
    }
}
