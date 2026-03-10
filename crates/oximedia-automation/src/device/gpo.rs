//! GPO (General Purpose Output) device implementation.

use crate::Result;
use std::collections::HashMap;
use tracing::info;

/// GPO device controller.
pub struct GpoDevice {
    port: String,
    pin_states: HashMap<u8, bool>,
}

impl GpoDevice {
    /// Create a new GPO device.
    pub async fn new(port: &str) -> Result<Self> {
        info!("Creating GPO device on port: {}", port);

        Ok(Self {
            port: port.to_string(),
            pin_states: HashMap::new(),
        })
    }

    /// Connect to the device.
    pub async fn connect(&mut self) -> Result<()> {
        info!("Connecting to GPO device on {}", self.port);

        // In a real implementation, this would open the hardware port
        // and initialize all pins to low

        for pin in 0..8 {
            self.pin_states.insert(pin, false);
        }

        Ok(())
    }

    /// Disconnect from the device.
    pub async fn disconnect(&mut self) -> Result<()> {
        info!("Disconnecting from GPO device on {}", self.port);

        // Reset all pins to low before disconnecting
        self.pin_states.clear();

        Ok(())
    }

    /// Trigger a pin (set high).
    pub async fn trigger(&mut self, pin: u8) -> Result<()> {
        info!("Triggering GPO pin {}", pin);

        self.pin_states.insert(pin, true);

        // In a real implementation, this would send the signal to hardware

        Ok(())
    }

    /// Clear a pin (set low).
    pub async fn clear(&mut self, pin: u8) -> Result<()> {
        info!("Clearing GPO pin {}", pin);

        self.pin_states.insert(pin, false);

        // In a real implementation, this would send the signal to hardware

        Ok(())
    }

    /// Pulse a pin (trigger then clear after duration).
    pub async fn pulse(&mut self, pin: u8, duration_ms: u64) -> Result<()> {
        info!("Pulsing GPO pin {} for {}ms", pin, duration_ms);

        self.trigger(pin).await?;
        tokio::time::sleep(tokio::time::Duration::from_millis(duration_ms)).await;
        self.clear(pin).await?;

        Ok(())
    }

    /// Get current state of a pin.
    pub fn get_pin_state(&self, pin: u8) -> bool {
        *self.pin_states.get(&pin).unwrap_or(&false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpo_device_creation() {
        let device = GpoDevice::new("/dev/gpio0").await;
        assert!(device.is_ok());
    }

    #[tokio::test]
    async fn test_gpo_trigger() {
        let mut device = GpoDevice::new("/dev/gpio0")
            .await
            .expect("new should succeed");
        device.connect().await.expect("operation should succeed");

        device.trigger(1).await.expect("operation should succeed");
        assert!(device.get_pin_state(1));

        device.clear(1).await.expect("operation should succeed");
        assert!(!device.get_pin_state(1));
    }
}
