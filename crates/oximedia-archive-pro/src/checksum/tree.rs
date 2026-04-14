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
                hex::encode(hasher.finalize())
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
                hex::encode(hasher.finalize())
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

    /// Generate a Merkle proof (authentication path) for a specific file.
    ///
    /// Returns a list of `MerkleProofStep` elements from the leaf up to the root.
    /// Each step contains the sibling hash and the direction indicating whether
    /// the sibling is on the left or right side.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not found in the tree.
    pub fn generate_proof(&self, path: &Path) -> Result<Vec<MerkleProofStep>> {
        let mut proof = Vec::new();
        if !Self::build_proof(&self.root, path, &mut proof) {
            return Err(Error::Metadata(format!(
                "File not found in Merkle tree: {}",
                path.display()
            )));
        }
        Ok(proof)
    }

    /// Recursively build a proof path. Returns `true` if the target was found.
    fn build_proof(node: &MerkleNode, target: &Path, proof: &mut Vec<MerkleProofStep>) -> bool {
        // Leaf: check if this is the target
        if node.is_leaf() {
            if let Some(ref node_path) = node.path {
                return node_path == target;
            }
            return false;
        }

        // Try left subtree
        if let Some(ref left) = node.left {
            if Self::build_proof(left, target, proof) {
                // Target is in the left subtree; sibling is on the right
                if let Some(ref right) = node.right {
                    proof.push(MerkleProofStep {
                        sibling_hash: right.hash.clone(),
                        direction: ProofDirection::Right,
                    });
                }
                return true;
            }
        }

        // Try right subtree
        if let Some(ref right) = node.right {
            if Self::build_proof(right, target, proof) {
                // Target is in the right subtree; sibling is on the left
                if let Some(ref left) = node.left {
                    proof.push(MerkleProofStep {
                        sibling_hash: left.hash.clone(),
                        direction: ProofDirection::Left,
                    });
                }
                return true;
            }
        }

        false
    }

    /// Verify a Merkle proof for a given leaf hash against the tree's root hash.
    ///
    /// This allows efficient partial verification: instead of re-hashing the
    /// entire tree, you only need the proof path (log N hashes) to confirm
    /// a single file's membership and integrity.
    #[must_use]
    pub fn verify_proof(&self, leaf_hash: &str, proof: &[MerkleProofStep]) -> bool {
        let mut current = leaf_hash.to_string();
        for step in proof {
            current = match step.direction {
                ProofDirection::Left => {
                    Self::combine_hashes(&step.sibling_hash, &current, self.algorithm)
                }
                ProofDirection::Right => {
                    Self::combine_hashes(&current, &step.sibling_hash, self.algorithm)
                }
            };
        }
        current == self.root.hash
    }

    /// Perform partial verification: verify a single file's integrity against
    /// the Merkle root without re-hashing the entire tree.
    ///
    /// Given a file path and its current hash, generates the proof path and
    /// verifies it against the stored root hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not found in the tree.
    pub fn verify_single_file(&self, path: &Path, current_hash: &str) -> Result<bool> {
        let proof = self.generate_proof(path)?;
        Ok(self.verify_proof(current_hash, &proof))
    }

    /// Detect which files have been corrupted by comparing against new hashes.
    ///
    /// Accepts a list of `(path, new_hash)` pairs and returns the paths of files
    /// whose proof verification fails (indicating corruption or tampering).
    pub fn detect_corrupted_files(&self, file_hashes: &[(PathBuf, String)]) -> Vec<PathBuf> {
        file_hashes
            .iter()
            .filter(|(path, hash)| {
                self.verify_single_file(path, hash)
                    .map(|ok| !ok)
                    .unwrap_or(true)
            })
            .map(|(path, _)| path.clone())
            .collect()
    }

    /// Collect all leaf nodes and their paths.
    #[must_use]
    pub fn leaf_entries(&self) -> Vec<(PathBuf, String)> {
        let mut entries = Vec::new();
        Self::collect_leaves(&self.root, &mut entries);
        entries
    }

    fn collect_leaves(node: &MerkleNode, entries: &mut Vec<(PathBuf, String)>) {
        if node.is_leaf() {
            if let Some(ref path) = node.path {
                entries.push((path.clone(), node.hash.clone()));
            }
            return;
        }
        if let Some(ref left) = node.left {
            Self::collect_leaves(left, entries);
        }
        if let Some(ref right) = node.right {
            Self::collect_leaves(right, entries);
        }
    }
}

/// Direction of a sibling node in a Merkle proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofDirection {
    /// Sibling is on the left
    Left,
    /// Sibling is on the right
    Right,
}

/// A single step in a Merkle proof (authentication path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProofStep {
    /// Hash of the sibling node at this level
    pub sibling_hash: String,
    /// Whether the sibling is on the left or right
    pub direction: ProofDirection,
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

    // ── Merkle proof tests ──────────────────────────────────────

    fn build_test_tree() -> MerkleTree {
        let files = vec![
            (PathBuf::from("a.txt"), "hash_a".to_string()),
            (PathBuf::from("b.txt"), "hash_b".to_string()),
            (PathBuf::from("c.txt"), "hash_c".to_string()),
            (PathBuf::from("d.txt"), "hash_d".to_string()),
        ];
        MerkleTree::build(files, ChecksumAlgorithm::Sha256).expect("operation should succeed")
    }

    #[test]
    fn test_generate_proof_existing_file() {
        let tree = build_test_tree();
        let proof = tree
            .generate_proof(Path::new("a.txt"))
            .expect("operation should succeed");
        assert!(!proof.is_empty());
    }

    #[test]
    fn test_generate_proof_nonexistent_file() {
        let tree = build_test_tree();
        let result = tree.generate_proof(Path::new("nonexistent.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_proof_valid() {
        let tree = build_test_tree();
        let path = Path::new("a.txt");
        let leaf_hash = tree
            .find_file(path)
            .expect("operation should succeed")
            .hash
            .clone();
        let proof = tree.generate_proof(path).expect("operation should succeed");
        assert!(tree.verify_proof(&leaf_hash, &proof));
    }

    #[test]
    fn test_verify_proof_all_leaves() {
        let tree = build_test_tree();
        let leaves = tree.leaf_entries();
        for (path, hash) in &leaves {
            let proof = tree.generate_proof(path).expect("operation should succeed");
            assert!(
                tree.verify_proof(hash, &proof),
                "Proof failed for {}",
                path.display()
            );
        }
    }

    #[test]
    fn test_verify_proof_wrong_hash() {
        let tree = build_test_tree();
        let path = Path::new("a.txt");
        let proof = tree.generate_proof(path).expect("operation should succeed");
        assert!(!tree.verify_proof("wrong_hash", &proof));
    }

    #[test]
    fn test_verify_single_file_correct() {
        let tree = build_test_tree();
        let path = Path::new("b.txt");
        let leaf_hash = tree
            .find_file(path)
            .expect("operation should succeed")
            .hash
            .clone();
        let result = tree
            .verify_single_file(path, &leaf_hash)
            .expect("operation should succeed");
        assert!(result);
    }

    #[test]
    fn test_verify_single_file_corrupted() {
        let tree = build_test_tree();
        let path = Path::new("c.txt");
        let result = tree
            .verify_single_file(path, "corrupted_hash")
            .expect("operation should succeed");
        assert!(!result);
    }

    #[test]
    fn test_detect_corrupted_files_none_corrupted() {
        let tree = build_test_tree();
        let entries = tree.leaf_entries();
        let corrupted = tree.detect_corrupted_files(&entries);
        assert!(corrupted.is_empty());
    }

    #[test]
    fn test_detect_corrupted_files_one_corrupted() {
        let tree = build_test_tree();
        let mut entries = tree.leaf_entries();
        // Corrupt the second entry
        entries[1].1 = "tampered".to_string();
        let corrupted = tree.detect_corrupted_files(&entries);
        assert_eq!(corrupted.len(), 1);
        assert_eq!(corrupted[0], PathBuf::from("b.txt"));
    }

    #[test]
    fn test_detect_corrupted_files_all_corrupted() {
        let tree = build_test_tree();
        let entries: Vec<(PathBuf, String)> = tree
            .leaf_entries()
            .into_iter()
            .map(|(p, _)| (p, "bad".to_string()))
            .collect();
        let corrupted = tree.detect_corrupted_files(&entries);
        assert_eq!(corrupted.len(), 4);
    }

    #[test]
    fn test_leaf_entries() {
        let tree = build_test_tree();
        let entries = tree.leaf_entries();
        assert_eq!(entries.len(), 4);
        let paths: Vec<&PathBuf> = entries.iter().map(|(p, _)| p).collect();
        assert!(paths.contains(&&PathBuf::from("a.txt")));
        assert!(paths.contains(&&PathBuf::from("d.txt")));
    }

    #[test]
    fn test_proof_single_file_tree() {
        let files = vec![(PathBuf::from("only.txt"), "only_hash".to_string())];
        let tree =
            MerkleTree::build(files, ChecksumAlgorithm::Blake3).expect("operation should succeed");
        let proof = tree
            .generate_proof(Path::new("only.txt"))
            .expect("operation should succeed");
        // Single-file tree: proof is empty (leaf is the root)
        assert!(proof.is_empty());
        assert!(tree.verify_proof("only_hash", &proof));
    }

    #[test]
    fn test_proof_odd_number_of_files() {
        let files = vec![
            (PathBuf::from("x.txt"), "hx".to_string()),
            (PathBuf::from("y.txt"), "hy".to_string()),
            (PathBuf::from("z.txt"), "hz".to_string()),
        ];
        let tree =
            MerkleTree::build(files, ChecksumAlgorithm::Sha256).expect("operation should succeed");
        for (path, hash) in tree.leaf_entries() {
            let proof = tree
                .generate_proof(&path)
                .expect("operation should succeed");
            assert!(tree.verify_proof(&hash, &proof));
        }
    }

    #[test]
    fn test_proof_blake3_algorithm() {
        let files = vec![
            (PathBuf::from("f1.bin"), "h1".to_string()),
            (PathBuf::from("f2.bin"), "h2".to_string()),
        ];
        let tree =
            MerkleTree::build(files, ChecksumAlgorithm::Blake3).expect("operation should succeed");
        let path = Path::new("f1.bin");
        let leaf_hash = tree
            .find_file(path)
            .expect("operation should succeed")
            .hash
            .clone();
        let proof = tree.generate_proof(path).expect("operation should succeed");
        assert!(tree.verify_proof(&leaf_hash, &proof));
    }
}
