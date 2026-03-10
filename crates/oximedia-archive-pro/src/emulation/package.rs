//! Emulation environment packaging

use super::EmulationPreparation;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Emulation package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmulationPackage {
    /// Package root
    pub root: PathBuf,
    /// Preparation info
    pub preparation: EmulationPreparation,
    /// Software files
    pub software_files: Vec<PathBuf>,
    /// Documentation files
    pub documentation_files: Vec<PathBuf>,
}

/// Emulation packager
pub struct EmulationPackager {
    root: PathBuf,
}

impl EmulationPackager {
    /// Create a new emulation packager
    ///
    /// # Errors
    ///
    /// Returns an error if package directory cannot be created
    pub fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join("software"))?;
        fs::create_dir_all(root.join("documentation"))?;
        fs::create_dir_all(root.join("configuration"))?;

        Ok(Self { root })
    }

    /// Create an emulation package
    ///
    /// # Errors
    ///
    /// Returns an error if package creation fails
    pub fn create_package(&self, preparation: EmulationPreparation) -> Result<EmulationPackage> {
        // Save preparation info
        let prep_file = self.root.join("emulation-prep.json");
        let json = serde_json::to_string_pretty(&preparation)
            .map_err(|e| crate::Error::Metadata(format!("JSON serialization failed: {e}")))?;
        fs::write(prep_file, json)?;

        // Create README
        self.create_readme(&preparation)?;

        Ok(EmulationPackage {
            root: self.root.clone(),
            preparation,
            software_files: Vec::new(),
            documentation_files: Vec::new(),
        })
    }

    fn create_readme(&self, prep: &EmulationPreparation) -> Result<()> {
        let mut readme = String::from("# Emulation Environment Package\n\n");
        readme.push_str(&format!("Format: {}\n\n", prep.format));
        readme.push_str("## Required Software\n\n");
        for sw in &prep.required_software {
            readme.push_str(&format!("- {sw}\n"));
        }
        readme.push_str("\n## Required Hardware\n\n");
        for hw in &prep.required_hardware {
            readme.push_str(&format!("- {hw}\n"));
        }
        readme.push_str("\n## Documentation\n\n");
        for doc in &prep.documentation {
            readme.push_str(&format!("- {doc}\n"));
        }

        fs::write(self.root.join("README.md"), readme)?;
        Ok(())
    }

    /// Add software file to package
    ///
    /// # Errors
    ///
    /// Returns an error if file cannot be added
    pub fn add_software(&self, source: &Path) -> Result<PathBuf> {
        let filename = source
            .file_name()
            .ok_or_else(|| crate::Error::Metadata("Invalid filename".to_string()))?;
        let dest = self.root.join("software").join(filename);
        fs::copy(source, &dest)?;
        Ok(dest)
    }

    /// Add documentation file to package
    ///
    /// # Errors
    ///
    /// Returns an error if file cannot be added
    pub fn add_documentation(&self, source: &Path) -> Result<PathBuf> {
        let filename = source
            .file_name()
            .ok_or_else(|| crate::Error::Metadata("Invalid filename".to_string()))?;
        let dest = self.root.join("documentation").join(filename);
        fs::copy(source, &dest)?;
        Ok(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_emulation_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let pkg_root = temp_dir.path().join("emulation");

        let packager = EmulationPackager::new(pkg_root.clone()).expect("operation should succeed");

        let prep = EmulationPreparation {
            format: "mkv".to_string(),
            required_software: vec!["FFmpeg".to_string()],
            required_hardware: vec!["x86_64".to_string()],
            config_files: Vec::new(),
            documentation: Vec::new(),
            timestamp: chrono::Utc::now(),
        };

        let package = packager
            .create_package(prep)
            .expect("operation should succeed");
        assert_eq!(package.root, pkg_root);
        assert!(pkg_root.join("README.md").exists());
        assert!(pkg_root.join("emulation-prep.json").exists());
    }
}
