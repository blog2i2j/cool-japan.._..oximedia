//! Facebook Gaming integration.

use crate::GamingResult;

/// Facebook Gaming integration.
pub struct FacebookIntegration {
    config: FacebookConfig,
}

/// Facebook configuration.
#[derive(Debug, Clone)]
pub struct FacebookConfig {
    /// Stream key
    pub stream_key: String,
    /// Stream title
    pub title: String,
    /// Description
    pub description: String,
}

impl FacebookIntegration {
    /// Create a new Facebook integration.
    #[must_use]
    pub fn new(config: FacebookConfig) -> Self {
        Self { config }
    }

    /// Update stream title.
    pub fn update_title(&mut self, title: String) -> GamingResult<()> {
        self.config.title = title;
        Ok(())
    }

    /// Update description.
    pub fn update_description(&mut self, description: String) -> GamingResult<()> {
        self.config.description = description;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facebook_integration() {
        let config = FacebookConfig {
            stream_key: "test_key".to_string(),
            title: "Test Stream".to_string(),
            description: "Test Description".to_string(),
        };
        let mut integration = FacebookIntegration::new(config);
        integration
            .update_title("New Title".to_string())
            .expect("update title should succeed");
    }
}
