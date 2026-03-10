//! Merkle tree for hierarchical checksum verification

use super::ChecksumAlgorithm;
use crate::{Error, Result};
use blake3::Hasher as Blake3Hasher;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// A node in a Merkle tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleNode {
    /// Hash of this node
    pub hash: String,
    /// Path (for leaf nodes)
    pub path: Option<PathBuf>,
    /// Left child
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left: Option<Box<MerkleNode>>,
    /// Right child
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right: Option<Box<MerkleNode>>,
}

impl MerkleNode {
    /// Create a new leaf node
    #[must_use]
    pub fn leaf(path: PathBuf, hash: String) -> Self {
        Self {
            hash,
            path: Some(path),
            left: None,
            right: None,
        }
    }

    /// Create a new internal node
    #[must_use]
    pub fn internal(hash: String, left: MerkleNode, right: MerkleNode) -> Self {
        Self {
            hash,
            path: None,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }

    /// Check if this is a leaf node
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Get the depth of the tree
    #[must_use]
    pub fn depth(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            1 + self
                .left
                .as_ref()
                .map_or(0, |l| l.depth())
                .max(self.right.as_ref().map_or(0, |r| r.depth()))
        }
    }

    /// Count the number of leaves
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            self.left.as_ref().map_or(0, |l| l.leaf_count())
                + self.right.as_ref().map_or(0, |r| r.leaf_count())
        }
    }
}

/// Merkle tree for hierarchical verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTree {
    /// Root node
    pub root: MerkleNode,
    /// Algorithm used
    pub algorithm: ChecksumAlgorithm,
    /// Timestamp of creation
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl MerkleTree {
    /// Build a Merkle tree from file hashes
    ///
    /// # Errors
    ///
    /// Returns an error if the input is empty
    pub fn build(files: Vec<(PathBuf, String)>, algorithm: ChecksumAlgorithm) -> Result<Self> {
        if files.is_empty() {
            return Err(Error::Metadata(
                "Cannot build Merkle tree from empty list".to_string(),
            ));
        }

        let mut nodes: Vec<MerkleNode> = files
            .into_iter()
            .map(|(path, hash)| MerkleNode::leaf(path, hash))
            .collect();

        // Build tree bottom-up
        while nodes.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in nodes.chunks(2) {
                if chunk.len() == 2 {
                    let left = chunk[0].clone();
                    let right = chunk[1].clone();
                    let combined_hash = Self::combine_hashes(&left.hash, &right.hash, algorithm);
                    next_level.push(MerkleNode::internal(combined_hash, left, right));
                } else {
                    // Odd number of nodes, promote last one
                    next_level.push(chunk[0].clone());
                }
            }

            nodes = next_level;
        }

        Ok(Self {
            root: nodes
                .into_iter()
                .next()
                .ok_or_else(|| Error::Metadata("Failed to build tree".to_string()))?,
            algorithm,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Combine two hashes into one
    fn combine_hashes(left: &str, right: &str, algorithm: ChecksumAlgorithm) -> String {
        let combined = format!("{left}{right}");

        match algorithm {
            ChecksumAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(combined.as_bytes());
                format!("{:x}", hasher.finalize())
            }
            ChecksumAlgorithm::Blake3 => {
                let mut hasher = Blake3Hasher::new();
                hasher.update(combined.as_bytes());
                hasher.finalize().to_hex().to_string()
            }
            _ => {
                // Default to SHA-256 for other algorithms
                let mut hasher = Sha256::new();
                hasher.update(combined.as_bytes());
                format!("{:x}", hasher.finalize())
            }
        }
    }

    /// Get the root hash
    #[must_use]
    pub fn root_hash(&self) -> &str {
        &self.root.hash
    }

    /// Get the depth of the tree
    #[must_use]
    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    /// Get the number of files in the tree
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.root.leaf_count()
    }

    /// Verify the tree structure
    #[must_use]
    pub fn verify_structure(&self) -> bool {
        Self::verify_node(&self.root, self.algorithm)
    }

    fn verify_node(node: &MerkleNode, algorithm: ChecksumAlgorithm) -> bool {
        if node.is_leaf() {
            true
        } else if let (Some(left), Some(right)) = (&node.left, &node.right) {
            let computed = Self::combine_hashes(&left.hash, &right.hash, algorithm);
            computed == node.hash
                && Self::verify_node(left, algorithm)
                && Self::verify_node(right, algorithm)
        } else {
            false
        }
    }

    /// Find a file in the tree
    #[must_use]
    pub fn find_file(&self, path: &Path) -> Option<&MerkleNode> {
        Self::find_in_node(&self.root, path)
    }

    fn find_in_node<'a>(node: &'a MerkleNode, path: &Path) -> Option<&'a MerkleNode> {
        if let Some(node_path) = &node.path {
            if node_path == path {
                return Some(node);
            }
        }

        if let Some(left) = &node.left {
            if let Some(found) = Self::find_in_node(left, path) {
                return Some(found);
            }
        }

        if let Some(right) = &node.right {
            if let Some(found) = Self::find_in_node(right, path) {
                return Some(found);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_merkle_node_leaf() {
        let node = MerkleNode::leaf(PathBuf::from("file.txt"), "abc123".to_string());
        assert!(node.is_leaf());
        assert_eq!(node.depth(), 1);
        assert_eq!(node.leaf_count(), 1);
    }

    #[test]
    fn test_merkle_node_internal() {
        let left = MerkleNode::leaf(PathBuf::from("a.txt"), "hash1".to_string());
        let right = MerkleNode::leaf(PathBuf::from("b.txt"), "hash2".to_string());
        let node = MerkleNode::internal("combined".to_string(), left, right);

        assert!(!node.is_leaf());
        assert_eq!(node.depth(), 2);
        assert_eq!(node.leaf_count(), 2);
    }

    #[test]
    fn test_build_merkle_tree() {
        let files = vec![
            (PathBuf::from("file1.txt"), "hash1".to_string()),
            (PathBuf::from("file2.txt"), "hash2".to_string()),
            (PathBuf::from("file3.txt"), "hash3".to_string()),
        ];

        let tree =
            MerkleTree::build(files, ChecksumAlgorithm::Sha256).expect("operation should succeed");
        assert_eq!(tree.file_count(), 3);
        assert!(tree.depth() >= 2);
    }

    #[test]
    fn test_verify_structure() {
        let files = vec![
            (PathBuf::from("a.txt"), "hash1".to_string()),
            (PathBuf::from("b.txt"), "hash2".to_string()),
        ];

        let tree =
            MerkleTree::build(files, ChecksumAlgorithm::Sha256).expect("operation should succeed");
        assert!(tree.verify_structure());
    }

    #[test]
    fn test_find_file() {
        let path1 = PathBuf::from("file1.txt");
        let path2 = PathBuf::from("file2.txt");

        let files = vec![
            (path1.clone(), "hash1".to_string()),
            (path2.clone(), "hash2".to_string()),
        ];

        let tree =
            MerkleTree::build(files, ChecksumAlgorithm::Sha256).expect("operation should succeed");

        assert!(tree.find_file(&path1).is_some());
        assert!(tree.find_file(&path2).is_some());
        assert!(tree.find_file(&PathBuf::from("nonexistent.txt")).is_none());
    }

    #[test]
    fn test_empty_tree() {
        let result = MerkleTree::build(vec![], ChecksumAlgorithm::Sha256);
        assert!(result.is_err());
    }
}
