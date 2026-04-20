//! Flame graph generation.

use crate::cpu::sample::{Sample, StackFrame};
use serde::{Deserialize, Serialize};

/// Flame graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameNode {
    /// Function name.
    pub name: String,

    /// Sample count.
    pub value: u64,

    /// Child nodes.
    pub children: Vec<FlameNode>,
}

/// Flame graph data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraphData {
    /// Root node.
    pub root: FlameNode,

    /// Total samples.
    pub total_samples: u64,
}

/// Flame graph generator.
#[derive(Debug)]
pub struct FlameGraphGenerator {
    samples: Vec<Sample>,
}

impl FlameGraphGenerator {
    /// Create a new flame graph generator.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Add samples.
    pub fn add_samples(&mut self, samples: &[Sample]) {
        self.samples.extend_from_slice(samples);
    }

    /// Generate flame graph data.
    pub fn generate(&self) -> FlameGraphData {
        let mut root = FlameNode {
            name: "root".to_string(),
            value: 0,
            children: Vec::new(),
        };

        for sample in &self.samples {
            self.add_sample_to_tree(&mut root, &sample.stack);
        }

        FlameGraphData {
            root,
            total_samples: self.samples.len() as u64,
        }
    }

    /// Add a sample to the flame graph tree.
    fn add_sample_to_tree(&self, node: &mut FlameNode, stack: &[StackFrame]) {
        if stack.is_empty() {
            return;
        }

        node.value += 1;

        let frame = &stack[0];
        let frame_name = frame.function.clone();

        // Find or create child node
        let child = node.children.iter_mut().find(|n| n.name == frame_name);

        let child = match child {
            Some(c) => c,
            None => {
                let new_child = FlameNode {
                    name: frame_name,
                    value: 0,
                    children: Vec::new(),
                };
                let idx = node.children.len();
                node.children.push(new_child);
                &mut node.children[idx]
            }
        };

        self.add_sample_to_tree(child, &stack[1..]);
    }

    /// Get total sample count.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

impl Default for FlameGraphGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flame_graph_generator() {
        let mut generator = FlameGraphGenerator::new();

        let mut sample = Sample::new(1, 50.0);
        sample.add_frame(StackFrame::new("func1".to_string(), 0x1000));
        sample.add_frame(StackFrame::new("func2".to_string(), 0x2000));

        generator.add_samples(&[sample]);
        assert_eq!(generator.sample_count(), 1);

        let data = generator.generate();
        assert_eq!(data.total_samples, 1);
        assert_eq!(data.root.value, 1);
    }

    #[test]
    fn test_flame_graph_merging() {
        let mut generator = FlameGraphGenerator::new();

        // Two samples with same stack prefix
        let mut sample1 = Sample::new(1, 50.0);
        sample1.add_frame(StackFrame::new("func1".to_string(), 0x1000));
        sample1.add_frame(StackFrame::new("func2".to_string(), 0x2000));

        let mut sample2 = Sample::new(1, 50.0);
        sample2.add_frame(StackFrame::new("func1".to_string(), 0x1000));
        sample2.add_frame(StackFrame::new("func3".to_string(), 0x3000));

        generator.add_samples(&[sample1, sample2]);

        let data = generator.generate();
        assert_eq!(data.root.children.len(), 1);
        assert_eq!(data.root.children[0].name, "func1");
        assert_eq!(data.root.children[0].value, 2);
        assert_eq!(data.root.children[0].children.len(), 2);
    }
}
