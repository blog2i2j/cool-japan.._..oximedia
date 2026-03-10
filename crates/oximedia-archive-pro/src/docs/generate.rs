//! Documentation package generation

use crate::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Documentation package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationPackage {
    /// Package root
    pub root: PathBuf,
    /// Technical documentation files
    pub technical_docs: Vec<PathBuf>,
    /// Descriptive documentation files
    pub descriptive_docs: Vec<PathBuf>,
    /// Generated timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Documentation generator
pub struct DocumentationGenerator {
    root: PathBuf,
}

impl DocumentationGenerator {
    /// Create a new documentation generator
    ///
    /// # Errors
    ///
    /// Returns an error if directories cannot be created
    pub fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join("technical"))?;
        fs::create_dir_all(root.join("descriptive"))?;
        Ok(Self { root })
    }

    /// Generate complete documentation package
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails
    pub fn generate(&self, file_path: &Path) -> Result<DocumentationPackage> {
        let mut package = DocumentationPackage {
            root: self.root.clone(),
            technical_docs: Vec::new(),
            descriptive_docs: Vec::new(),
            timestamp: chrono::Utc::now(),
        };

        // Generate technical documentation
        let tech_doc = self.generate_technical_doc(file_path)?;
        package.technical_docs.push(tech_doc);

        // Generate descriptive documentation
        let desc_doc = self.generate_descriptive_doc(file_path)?;
        package.descriptive_docs.push(desc_doc);

        // Generate index
        self.generate_index(&package)?;

        Ok(package)
    }

    fn generate_technical_doc(&self, file_path: &Path) -> Result<PathBuf> {
        let metadata = fs::metadata(file_path)?;
        let mut doc = String::from("# Technical Documentation\n\n");

        doc.push_str(&format!("**File:** {}\n\n", file_path.display()));
        doc.push_str(&format!("**Size:** {} bytes\n\n", metadata.len()));
        doc.push_str(&format!("**Generated:** {}\n\n", chrono::Utc::now()));

        if let Some(ext) = file_path.extension() {
            doc.push_str(&format!("**Format:** {}\n\n", ext.to_string_lossy()));
        }

        let doc_path = self.root.join("technical").join("technical.md");
        fs::write(&doc_path, doc)?;
        Ok(doc_path)
    }

    fn generate_descriptive_doc(&self, file_path: &Path) -> Result<PathBuf> {
        let mut doc = String::from("# Descriptive Metadata\n\n");

        if let Some(name) = file_path.file_name() {
            doc.push_str(&format!("**Title:** {}\n\n", name.to_string_lossy()));
        }

        doc.push_str(&format!(
            "**Date Created:** {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d")
        ));

        let doc_path = self.root.join("descriptive").join("descriptive.md");
        fs::write(&doc_path, doc)?;
        Ok(doc_path)
    }

    fn generate_index(&self, package: &DocumentationPackage) -> Result<()> {
        let mut index = String::from("# Preservation Documentation Index\n\n");

        index.push_str("## Technical Documentation\n\n");
        for doc in &package.technical_docs {
            if let Some(name) = doc.file_name() {
                index.push_str(&format!(
                    "- [{}](technical/{})\n",
                    name.to_string_lossy(),
                    name.to_string_lossy()
                ));
            }
        }

        index.push_str("\n## Descriptive Documentation\n\n");
        for doc in &package.descriptive_docs {
            if let Some(name) = doc.file_name() {
                index.push_str(&format!(
                    "- [{}](descriptive/{})\n",
                    name.to_string_lossy(),
                    name.to_string_lossy()
                ));
            }
        }

        fs::write(self.root.join("INDEX.md"), index)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_generate_documentation() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let doc_root = temp_dir.path().join("docs");

        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator =
            DocumentationGenerator::new(doc_root.clone()).expect("operation should succeed");
        let package = generator
            .generate(file.path())
            .expect("operation should succeed");

        assert!(!package.technical_docs.is_empty());
        assert!(!package.descriptive_docs.is_empty());
        assert!(doc_root.join("INDEX.md").exists());
    }
}
