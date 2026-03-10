// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Cloud provider integration for hybrid rendering.

use crate::error::Result;
use crate::worker::Worker;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;

/// Cloud provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudProvider {
    /// Amazon Web Services
    AWS,
    /// Microsoft Azure
    Azure,
    /// Google Cloud Platform
    GCP,
    /// Custom provider
    Custom,
}

/// Instance type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceType {
    /// Instance ID
    pub id: String,
    /// CPU cores
    pub cpu_cores: u32,
    /// RAM in GB
    pub ram_gb: u32,
    /// GPU available
    pub has_gpu: bool,
    /// GPU model
    pub gpu_model: Option<String>,
    /// Hourly cost
    pub hourly_cost: f64,
    /// On-demand or spot
    pub spot: bool,
}

/// Cloud configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Provider
    pub provider: CloudProvider,
    /// Region
    pub region: String,
    /// Access credentials
    pub credentials: CloudCredentials,
    /// Enable auto-scaling
    pub enable_auto_scaling: bool,
    /// Minimum instances
    pub min_instances: u32,
    /// Maximum instances
    pub max_instances: u32,
    /// Target utilization (0.0 to 1.0)
    pub target_utilization: f64,
}

/// Cloud credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudCredentials {
    /// Access key ID
    pub access_key_id: String,
    /// Secret access key
    pub secret_access_key: String,
    /// Additional parameters
    pub params: HashMap<String, String>,
}

/// Cloud instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudInstance {
    /// Instance ID
    pub id: String,
    /// Instance type
    pub instance_type: InstanceType,
    /// Public IP address
    pub public_ip: IpAddr,
    /// Private IP address
    pub private_ip: IpAddr,
    /// State
    pub state: InstanceState,
    /// Launch time
    pub launch_time: chrono::DateTime<chrono::Utc>,
}

/// Instance state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstanceState {
    /// Starting
    Pending,
    /// Running
    Running,
    /// Stopping
    Stopping,
    /// Stopped
    Stopped,
    /// Terminated
    Terminated,
}

/// Cloud manager
pub struct CloudManager {
    config: CloudConfig,
    instances: HashMap<String, CloudInstance>,
    #[allow(dead_code)]
    workers: HashMap<String, Worker>,
}

impl CloudManager {
    /// Create a new cloud manager
    #[must_use]
    pub fn new(config: CloudConfig) -> Self {
        Self {
            config,
            instances: HashMap::new(),
            workers: HashMap::new(),
        }
    }

    /// Launch instances
    pub async fn launch_instances(
        &mut self,
        count: u32,
        instance_type: InstanceType,
    ) -> Result<Vec<String>> {
        let mut instance_ids = Vec::new();

        for _ in 0..count {
            let instance_id = format!("i-{}", uuid::Uuid::new_v4());

            let instance = CloudInstance {
                id: instance_id.clone(),
                instance_type: instance_type.clone(),
                public_ip: std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED),
                private_ip: std::net::IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 1)),
                state: InstanceState::Pending,
                launch_time: chrono::Utc::now(),
            };

            self.instances.insert(instance_id.clone(), instance);
            instance_ids.push(instance_id);
        }

        Ok(instance_ids)
    }

    /// Terminate instances
    pub async fn terminate_instances(&mut self, instance_ids: Vec<String>) -> Result<()> {
        for instance_id in instance_ids {
            if let Some(instance) = self.instances.get_mut(&instance_id) {
                instance.state = InstanceState::Terminated;
            }
            self.instances.remove(&instance_id);
        }
        Ok(())
    }

    /// Get instance by ID
    #[must_use]
    pub fn get_instance(&self, instance_id: &str) -> Option<&CloudInstance> {
        self.instances.get(instance_id)
    }

    /// List all instances
    #[must_use]
    pub fn list_instances(&self) -> Vec<&CloudInstance> {
        self.instances.values().collect()
    }

    /// Scale up
    pub async fn scale_up(&mut self, count: u32) -> Result<()> {
        let instance_type = InstanceType {
            id: "c5.2xlarge".to_string(),
            cpu_cores: 8,
            ram_gb: 16,
            has_gpu: false,
            gpu_model: None,
            hourly_cost: 0.34,
            spot: false,
        };

        self.launch_instances(count, instance_type).await?;
        Ok(())
    }

    /// Scale down
    pub async fn scale_down(&mut self, count: u32) -> Result<()> {
        let instances_to_terminate: Vec<String> = self
            .instances
            .keys()
            .take(count as usize)
            .cloned()
            .collect();

        self.terminate_instances(instances_to_terminate).await?;
        Ok(())
    }

    /// Auto-scale based on utilization
    pub async fn auto_scale(&mut self, current_utilization: f64) -> Result<()> {
        if !self.config.enable_auto_scaling {
            return Ok(());
        }

        let current_instances = self.instances.len() as u32;

        if current_utilization > self.config.target_utilization
            && current_instances < self.config.max_instances
        {
            // Scale up
            let scale_up_count =
                ((current_utilization - self.config.target_utilization) * 10.0) as u32;
            let scale_up_count = scale_up_count.min(self.config.max_instances - current_instances);
            self.scale_up(scale_up_count).await?;
        } else if current_utilization < self.config.target_utilization * 0.5
            && current_instances > self.config.min_instances
        {
            // Scale down
            let scale_down_count =
                ((self.config.target_utilization - current_utilization) * 5.0) as u32;
            let scale_down_count =
                scale_down_count.min(current_instances - self.config.min_instances);
            self.scale_down(scale_down_count).await?;
        }

        Ok(())
    }

    /// Calculate cloud cost
    #[must_use]
    pub fn calculate_cost(&self, hours: f64) -> f64 {
        self.instances
            .values()
            .map(|i| i.instance_type.hourly_cost * hours)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> CloudConfig {
        CloudConfig {
            provider: CloudProvider::AWS,
            region: "us-west-2".to_string(),
            credentials: CloudCredentials {
                access_key_id: "test".to_string(),
                secret_access_key: "test".to_string(),
                params: HashMap::new(),
            },
            enable_auto_scaling: true,
            min_instances: 1,
            max_instances: 10,
            target_utilization: 0.7,
        }
    }

    #[tokio::test]
    async fn test_cloud_manager_creation() {
        let config = create_test_config();
        let manager = CloudManager::new(config);
        assert_eq!(manager.instances.len(), 0);
    }

    #[tokio::test]
    async fn test_launch_instances() -> Result<()> {
        let config = create_test_config();
        let mut manager = CloudManager::new(config);

        let instance_type = InstanceType {
            id: "c5.2xlarge".to_string(),
            cpu_cores: 8,
            ram_gb: 16,
            has_gpu: false,
            gpu_model: None,
            hourly_cost: 0.34,
            spot: false,
        };

        let instance_ids = manager.launch_instances(2, instance_type).await?;
        assert_eq!(instance_ids.len(), 2);
        assert_eq!(manager.instances.len(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_terminate_instances() -> Result<()> {
        let config = create_test_config();
        let mut manager = CloudManager::new(config);

        let instance_type = InstanceType {
            id: "c5.2xlarge".to_string(),
            cpu_cores: 8,
            ram_gb: 16,
            has_gpu: false,
            gpu_model: None,
            hourly_cost: 0.34,
            spot: false,
        };

        let instance_ids = manager.launch_instances(2, instance_type).await?;
        manager.terminate_instances(instance_ids).await?;

        assert_eq!(manager.instances.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_scale_up() -> Result<()> {
        let config = create_test_config();
        let mut manager = CloudManager::new(config);

        manager.scale_up(3).await?;
        assert_eq!(manager.instances.len(), 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_scale_down() -> Result<()> {
        let config = create_test_config();
        let mut manager = CloudManager::new(config);

        manager.scale_up(5).await?;
        manager.scale_down(2).await?;
        assert_eq!(manager.instances.len(), 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_auto_scale() -> Result<()> {
        let config = create_test_config();
        let mut manager = CloudManager::new(config);

        // High utilization - should scale up
        manager.auto_scale(0.9).await?;
        let count_after_scale_up = manager.instances.len();
        assert!(count_after_scale_up > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_calculate_cost() -> Result<()> {
        let config = create_test_config();
        let mut manager = CloudManager::new(config);

        let instance_type = InstanceType {
            id: "c5.2xlarge".to_string(),
            cpu_cores: 8,
            ram_gb: 16,
            has_gpu: false,
            gpu_model: None,
            hourly_cost: 0.34,
            spot: false,
        };

        manager.launch_instances(2, instance_type).await?;

        let cost = manager.calculate_cost(1.0); // 1 hour
        assert!((cost - 0.68).abs() < 0.01);

        Ok(())
    }
}
