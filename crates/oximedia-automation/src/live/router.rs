//! Video router control automation.

use crate::{AutomationError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Router crosspoint connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crosspoint {
    /// Input number
    pub input: usize,
    /// Output number
    pub output: usize,
}

/// Router automation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Number of inputs
    pub num_inputs: usize,
    /// Number of outputs
    pub num_outputs: usize,
    /// Router IP address
    pub ip_address: String,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            num_inputs: 32,
            num_outputs: 32,
            ip_address: "192.168.1.100".to_string(),
        }
    }
}

/// Video router automation.
pub struct RouterAutomation {
    config: RouterConfig,
    crosspoints: HashMap<usize, usize>,
}

impl RouterAutomation {
    /// Create a new router automation.
    pub fn new(config: RouterConfig) -> Self {
        info!(
            "Creating router automation: {}x{} at {}",
            config.num_inputs, config.num_outputs, config.ip_address
        );

        Self {
            config,
            crosspoints: HashMap::new(),
        }
    }

    /// Set a crosspoint (connect input to output).
    pub fn set_crosspoint(&mut self, input: usize, output: usize) -> Result<()> {
        if input >= self.config.num_inputs {
            return Err(AutomationError::LiveSwitching(format!(
                "Invalid input: {} (max: {})",
                input,
                self.config.num_inputs - 1
            )));
        }

        if output >= self.config.num_outputs {
            return Err(AutomationError::LiveSwitching(format!(
                "Invalid output: {} (max: {})",
                output,
                self.config.num_outputs - 1
            )));
        }

        debug!("Setting crosspoint: input {} -> output {}", input, output);
        self.crosspoints.insert(output, input);

        Ok(())
    }

    /// Get current input for an output.
    pub fn get_crosspoint(&self, output: usize) -> Option<usize> {
        self.crosspoints.get(&output).copied()
    }

    /// Clear all crosspoints.
    pub fn clear_all(&mut self) {
        info!("Clearing all crosspoints");
        self.crosspoints.clear();
    }

    /// Get all active crosspoints.
    pub fn get_all_crosspoints(&self) -> Vec<Crosspoint> {
        self.crosspoints
            .iter()
            .map(|(&output, &input)| Crosspoint { input, output })
            .collect()
    }

    /// Perform automated salvo (multiple simultaneous changes).
    pub fn salvo(&mut self, crosspoints: &[Crosspoint]) -> Result<()> {
        info!("Performing salvo with {} crosspoints", crosspoints.len());

        for cp in crosspoints {
            self.set_crosspoint(cp.input, cp.output)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let config = RouterConfig::default();
        let router = RouterAutomation::new(config);
        assert_eq!(router.get_all_crosspoints().len(), 0);
    }

    #[test]
    fn test_set_crosspoint() {
        let config = RouterConfig::default();
        let mut router = RouterAutomation::new(config);

        router
            .set_crosspoint(1, 5)
            .expect("set_crosspoint should succeed");
        assert_eq!(router.get_crosspoint(5), Some(1));
    }

    #[test]
    fn test_invalid_crosspoint() {
        let config = RouterConfig {
            num_inputs: 4,
            num_outputs: 4,
            ..Default::default()
        };
        let mut router = RouterAutomation::new(config);

        let result = router.set_crosspoint(10, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_salvo() {
        let config = RouterConfig::default();
        let mut router = RouterAutomation::new(config);

        let crosspoints = vec![
            Crosspoint {
                input: 1,
                output: 5,
            },
            Crosspoint {
                input: 2,
                output: 6,
            },
            Crosspoint {
                input: 3,
                output: 7,
            },
        ];

        router.salvo(&crosspoints).expect("salvo should succeed");
        assert_eq!(router.get_crosspoint(5), Some(1));
        assert_eq!(router.get_crosspoint(6), Some(2));
        assert_eq!(router.get_crosspoint(7), Some(3));
    }
}
