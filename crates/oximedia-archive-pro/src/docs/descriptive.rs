//! Descriptive documentation generation

use crate::Result;
use std::path::Path;

/// Descriptive documentation generator
pub struct DescriptiveDocGenerator;

impl Default for DescriptiveDocGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl DescriptiveDocGenerator {
    /// Create a new descriptive documentation generator
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Generate descriptive documentation
    ///
    /// # Errors
    ///
    /// Returns an error if documentation cannot be generated
    pub fn generate(
        &self,
        file_path: &Path,
        title: Option<&str>,
        creator: Option<&str>,
    ) -> Result<String> {
        let mut doc = String::from("# Descriptive Metadata\n\n");

        doc.push_str("## Identification\n\n");

        let title = title.or_else(|| file_path.file_stem().and_then(|s| s.to_str()));
        if let Some(t) = title {
            doc.push_str(&format!("**Title**: {t}\n\n"));
        }

        if let Some(c) = creator {
            doc.push_str(&format!("**Creator**: {c}\n\n"));
        }

        doc.push_str(&format!(
            "**Date**: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d")
        ));

        doc.push_str("## Description\n\n");
        doc.push_str("*[Description would be provided by curator]*\n\n");

        doc.push_str("## Subject Keywords\n\n");
        doc.push_str("*[Subject keywords would be assigned]*\n\n");

        doc.push_str("## Rights\n\n");
        doc.push_str("*[Rights information would be specified]*\n\n");

        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_generate_descriptive_doc() {
        let generator = DescriptiveDocGenerator::new();
        let doc = generator
            .generate(
                &PathBuf::from("test_video.mkv"),
                Some("Test Video"),
                Some("Test Creator"),
            )
            .expect("operation should succeed");

        assert!(doc.contains("Descriptive Metadata"));
        assert!(doc.contains("Test Video"));
        assert!(doc.contains("Test Creator"));
    }
}
