//! Territory-based royalty rate adjustments
//!
//! Different territories have different royalty rate multipliers based on
//! market conditions, local regulations, and historical industry practices.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Territory identifier for royalty calculations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Territory {
    /// United States
    US,
    /// United Kingdom
    UK,
    /// European Union (aggregate)
    EU,
    /// Japan
    Japan,
    /// Canada
    Canada,
    /// Australia
    Australia,
    /// South Korea
    SouthKorea,
    /// Brazil
    Brazil,
    /// Mexico
    Mexico,
    /// China
    China,
    /// India
    India,
    /// Russia
    Russia,
    /// Argentina
    Argentina,
    /// South Africa
    SouthAfrica,
    /// Custom territory identified by ISO 3166-1 alpha-2 code
    Custom(String),
    /// Worldwide (all territories)
    Worldwide,
}

impl Territory {
    /// Get royalty rate multiplier for this territory.
    ///
    /// Returns a multiplier relative to the US baseline (1.0).
    /// Values above 1.0 indicate higher royalties, below 1.0 indicate lower.
    ///
    /// # Examples
    /// ```
    /// use oximedia_rights::royalty::territory::Territory;
    ///
    /// assert!((Territory::US.rate_multiplier() - 1.0).abs() < f64::EPSILON);
    /// assert!(Territory::Japan.rate_multiplier() > 1.0);
    /// assert!(Territory::India.rate_multiplier() < 1.0);
    /// ```
    pub fn rate_multiplier(&self) -> f64 {
        match self {
            // Tier 1: Full rate markets
            Territory::US => 1.0,
            Territory::UK => 1.0,
            Territory::EU => 1.0,
            Territory::Canada => 0.95,
            Territory::Australia => 0.90,

            // Tier 2: Premium markets (higher collection rates)
            Territory::Japan => 1.2,
            Territory::SouthKorea => 0.85,

            // Tier 3: Mid-tier markets
            Territory::Brazil => 0.55,
            Territory::Mexico => 0.50,
            Territory::Russia => 0.45,
            Territory::Argentina => 0.40,
            Territory::China => 0.60,

            // Tier 4: Developing markets
            Territory::India => 0.35,
            Territory::SouthAfrica => 0.45,

            // Custom territories default to mid-range
            Territory::Custom(_) => 0.50,

            // Worldwide uses a blended rate
            Territory::Worldwide => 0.75,
        }
    }

    /// Get the currency code typically used for this territory
    pub fn currency_code(&self) -> &'static str {
        match self {
            Territory::US => "USD",
            Territory::UK => "GBP",
            Territory::EU => "EUR",
            Territory::Japan => "JPY",
            Territory::Canada => "CAD",
            Territory::Australia => "AUD",
            Territory::SouthKorea => "KRW",
            Territory::Brazil => "BRL",
            Territory::Mexico => "MXN",
            Territory::China => "CNY",
            Territory::India => "INR",
            Territory::Russia => "RUB",
            Territory::Argentina => "ARS",
            Territory::SouthAfrica => "ZAR",
            Territory::Custom(_) | Territory::Worldwide => "USD",
        }
    }

    /// Parse a territory from an ISO 3166-1 alpha-2 country code
    ///
    /// # Examples
    /// ```
    /// use oximedia_rights::royalty::territory::Territory;
    ///
    /// assert_eq!(Territory::from_country_code("US"), Territory::US);
    /// assert_eq!(Territory::from_country_code("JP"), Territory::Japan);
    /// assert_eq!(Territory::from_country_code("XX"), Territory::Custom("XX".to_string()));
    /// ```
    pub fn from_country_code(code: &str) -> Self {
        match code.to_uppercase().as_str() {
            "US" => Territory::US,
            "GB" => Territory::UK,
            // EU member states map to EU territory
            "DE" | "FR" | "IT" | "ES" | "NL" | "BE" | "PL" | "SE" | "AT" | "CH" | "NO" | "DK"
            | "FI" | "PT" | "GR" | "CZ" | "IE" | "HU" | "RO" => Territory::EU,
            "JP" => Territory::Japan,
            "CA" => Territory::Canada,
            "AU" => Territory::Australia,
            "KR" => Territory::SouthKorea,
            "BR" => Territory::Brazil,
            "MX" => Territory::Mexico,
            "CN" => Territory::China,
            "IN" => Territory::India,
            "RU" => Territory::Russia,
            "AR" => Territory::Argentina,
            "ZA" => Territory::SouthAfrica,
            other => Territory::Custom(other.to_string()),
        }
    }

    /// Get a human-readable name for this territory
    pub fn display_name(&self) -> String {
        match self {
            Territory::US => "United States".to_string(),
            Territory::UK => "United Kingdom".to_string(),
            Territory::EU => "European Union".to_string(),
            Territory::Japan => "Japan".to_string(),
            Territory::Canada => "Canada".to_string(),
            Territory::Australia => "Australia".to_string(),
            Territory::SouthKorea => "South Korea".to_string(),
            Territory::Brazil => "Brazil".to_string(),
            Territory::Mexico => "Mexico".to_string(),
            Territory::China => "China".to_string(),
            Territory::India => "India".to_string(),
            Territory::Russia => "Russia".to_string(),
            Territory::Argentina => "Argentina".to_string(),
            Territory::SouthAfrica => "South Africa".to_string(),
            Territory::Custom(code) => format!("Territory({code})"),
            Territory::Worldwide => "Worldwide".to_string(),
        }
    }
}

/// A table mapping territories to custom rate multipliers.
/// Used when standard territory multipliers need to be overridden per-deal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerritoryRateTable {
    /// Custom overrides: territory -> multiplier
    overrides: HashMap<String, f64>,
}

impl TerritoryRateTable {
    /// Create an empty rate table (will use Territory defaults)
    pub fn new() -> Self {
        Self {
            overrides: HashMap::new(),
        }
    }

    /// Set a custom multiplier for a specific territory
    ///
    /// # Arguments
    /// * `territory` - The territory to set a rate for
    /// * `multiplier` - The rate multiplier (must be positive)
    pub fn set_rate(&mut self, territory: Territory, multiplier: f64) -> Result<(), String> {
        if multiplier <= 0.0 {
            return Err(format!(
                "Territory rate multiplier must be positive, got {multiplier}"
            ));
        }
        let key = format!("{territory:?}");
        self.overrides.insert(key, multiplier);
        Ok(())
    }

    /// Get the effective rate multiplier for a territory.
    /// Returns the custom override if set, otherwise the territory's default.
    pub fn get_multiplier(&self, territory: &Territory) -> f64 {
        let key = format!("{territory:?}");
        self.overrides
            .get(&key)
            .copied()
            .unwrap_or_else(|| territory.rate_multiplier())
    }
}

impl Default for TerritoryRateTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_territory_multipliers() {
        // Tier 1 markets at 1.0
        assert!((Territory::US.rate_multiplier() - 1.0).abs() < f64::EPSILON);
        assert!((Territory::UK.rate_multiplier() - 1.0).abs() < f64::EPSILON);
        assert!((Territory::EU.rate_multiplier() - 1.0).abs() < f64::EPSILON);

        // Japan premium
        assert!(Territory::Japan.rate_multiplier() > 1.0);
        assert!((Territory::Japan.rate_multiplier() - 1.2).abs() < f64::EPSILON);

        // Developing markets below 0.7
        assert!(Territory::India.rate_multiplier() < 0.5);
        assert!(Territory::Brazil.rate_multiplier() < 0.7);
    }

    #[test]
    fn test_from_country_code() {
        assert_eq!(Territory::from_country_code("US"), Territory::US);
        assert_eq!(Territory::from_country_code("us"), Territory::US);
        assert_eq!(Territory::from_country_code("JP"), Territory::Japan);
        assert_eq!(Territory::from_country_code("GB"), Territory::UK);
        assert_eq!(Territory::from_country_code("DE"), Territory::EU);
        assert_eq!(Territory::from_country_code("FR"), Territory::EU);
        assert_eq!(
            Territory::from_country_code("XX"),
            Territory::Custom("XX".to_string())
        );
    }

    #[test]
    fn test_rate_table_override() {
        let mut table = TerritoryRateTable::new();
        // Default for India is 0.35
        let default_india = table.get_multiplier(&Territory::India);
        assert!((default_india - 0.35).abs() < f64::EPSILON);

        // Override India to 0.5
        table
            .set_rate(Territory::India, 0.5)
            .expect("rights test operation should succeed");
        let overridden = table.get_multiplier(&Territory::India);
        assert!((overridden - 0.5).abs() < f64::EPSILON);

        // US remains unchanged
        let us_rate = table.get_multiplier(&Territory::US);
        assert!((us_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rate_table_invalid_multiplier() {
        let mut table = TerritoryRateTable::new();
        let result = table.set_rate(Territory::US, -0.5);
        assert!(result.is_err());
        let result2 = table.set_rate(Territory::US, 0.0);
        assert!(result2.is_err());
    }

    #[test]
    fn test_currency_codes() {
        assert_eq!(Territory::US.currency_code(), "USD");
        assert_eq!(Territory::UK.currency_code(), "GBP");
        assert_eq!(Territory::Japan.currency_code(), "JPY");
        assert_eq!(Territory::EU.currency_code(), "EUR");
    }

    #[test]
    fn test_display_names() {
        assert_eq!(Territory::US.display_name(), "United States");
        assert_eq!(Territory::Worldwide.display_name(), "Worldwide");
        assert_eq!(
            Territory::Custom("XX".to_string()).display_name(),
            "Territory(XX)"
        );
    }
}
