//! Vorbis codebook structures.
//!
//! Codebooks are used for entropy coding in Vorbis. They consist of:
//! - A Huffman tree for codeword decoding
//! - An optional vector quantization (VQ) lookup table
//!
//! # Codebook Types
//!
//! - Type 0: Scalar codebook (no VQ)
//! - Type 1: Lattice VQ (implicit values)
//! - Type 2: Tessellated VQ (explicit values)

#![forbid(unsafe_code)]

use crate::AudioError;

/// Codebook entry containing codeword and optional VQ value.
#[derive(Debug, Clone, Default)]
pub struct CodebookEntry {
    /// Codeword length in bits.
    pub length: u8,
    /// Codeword value.
    pub codeword: u32,
    /// Entry used flag.
    pub used: bool,
    /// VQ lookup values (if applicable).
    pub values: Vec<f32>,
}

impl CodebookEntry {
    /// Create a new unused entry.
    #[must_use]
    pub fn unused() -> Self {
        Self {
            length: 0,
            codeword: 0,
            used: false,
            values: Vec::new(),
        }
    }

    /// Create a new used entry with given length.
    #[must_use]
    pub fn new(length: u8) -> Self {
        Self {
            length,
            codeword: 0,
            used: length > 0,
            values: Vec::new(),
        }
    }

    /// Check if this entry is valid for decoding.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.used && self.length > 0
    }
}

/// Huffman tree node.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub enum HuffmanNode {
    /// Internal node with left and right children.
    Internal {
        /// Left child (bit 0).
        left: Box<HuffmanNode>,
        /// Right child (bit 1).
        right: Box<HuffmanNode>,
    },
    /// Leaf node with entry index.
    Leaf(usize),
    /// Empty node (no entry).
    #[default]
    Empty,
}

/// Huffman tree for codeword decoding.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct HuffmanTree {
    /// Root node.
    root: HuffmanNode,
    /// Maximum code length.
    max_length: u8,
    /// Number of entries.
    entry_count: usize,
}

impl HuffmanTree {
    /// Create a new empty Huffman tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: HuffmanNode::Empty,
            max_length: 0,
            entry_count: 0,
        }
    }

    /// Build Huffman tree from entry lengths.
    ///
    /// # Errors
    ///
    /// Returns error if lengths are invalid.
    pub fn build(lengths: &[u8]) -> Result<Self, AudioError> {
        if lengths.is_empty() {
            return Ok(Self::new());
        }

        let max_length = *lengths.iter().max().unwrap_or(&0);
        if max_length > 32 {
            return Err(AudioError::InvalidData("Code length too long".into()));
        }

        // Validate using Kraft inequality
        let kraft_sum: u64 = lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u64 << (max_length - l))
            .sum();

        if kraft_sum > (1u64 << max_length) {
            return Err(AudioError::InvalidData(
                "Invalid Huffman code lengths".into(),
            ));
        }

        let entry_count = lengths.len();

        Ok(Self {
            root: HuffmanNode::Empty, // Skeleton: actual tree building is complex
            max_length,
            entry_count,
        })
    }

    /// Get maximum code length.
    #[must_use]
    pub fn max_length(&self) -> u8 {
        self.max_length
    }

    /// Get number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Decode a single symbol (skeleton).
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    #[allow(dead_code)]
    pub fn decode(&self, _bits: &mut dyn Iterator<Item = bool>) -> Result<usize, AudioError> {
        // Skeleton implementation
        Err(AudioError::InvalidData(
            "Huffman decode not implemented".into(),
        ))
    }
}

/// Codebook lookup type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LookupType {
    /// No lookup (scalar codebook).
    #[default]
    None,
    /// Type 1: Lattice VQ.
    Lattice,
    /// Type 2: Tessellated VQ.
    Tessellated,
}

impl LookupType {
    /// Create from raw value.
    #[must_use]
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(LookupType::None),
            1 => Some(LookupType::Lattice),
            2 => Some(LookupType::Tessellated),
            _ => None,
        }
    }

    /// Check if this type has VQ values.
    #[must_use]
    pub fn has_lookup(self) -> bool {
        self != LookupType::None
    }
}

/// Vorbis codebook.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct Codebook {
    /// Codebook identifier/index.
    pub id: usize,
    /// Number of entries.
    pub entries: usize,
    /// Entry dimensions (for VQ).
    pub dimensions: u16,
    /// Huffman tree for decoding.
    pub tree: HuffmanTree,
    /// Lookup type.
    pub lookup_type: LookupType,
    /// Minimum value for VQ.
    pub minimum_value: f32,
    /// Delta value for VQ.
    pub delta_value: f32,
    /// Value bits for VQ.
    pub value_bits: u8,
    /// Sequence flag for VQ.
    pub sequence_p: bool,
    /// Multiplicands for VQ lookup.
    pub multiplicands: Vec<u32>,
    /// Codebook entries.
    pub entry_list: Vec<CodebookEntry>,
}

impl Codebook {
    /// Create a new empty codebook.
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    /// Parse codebook from bit reader (skeleton).
    ///
    /// # Errors
    ///
    /// Returns error if codebook is invalid.
    pub fn parse(id: usize, data: &[u8]) -> Result<Self, AudioError> {
        // Check sync pattern (0x564342 = "BCV")
        if data.len() < 10 {
            return Err(AudioError::InvalidData("Codebook data too short".into()));
        }

        // Skeleton: actual parsing requires bit-level reading
        Ok(Self::new(id))
    }

    /// Get number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries
    }

    /// Get dimensions.
    #[must_use]
    pub fn dimensions(&self) -> u16 {
        self.dimensions
    }

    /// Decode scalar value (skeleton).
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    #[allow(dead_code)]
    pub fn decode_scalar(&self, _bits: &mut dyn Iterator<Item = bool>) -> Result<u32, AudioError> {
        Err(AudioError::InvalidData(
            "Codebook decode not implemented".into(),
        ))
    }

    /// Decode VQ vector (skeleton).
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    #[allow(dead_code)]
    pub fn decode_vq(&self, _bits: &mut dyn Iterator<Item = bool>) -> Result<Vec<f32>, AudioError> {
        if !self.lookup_type.has_lookup() {
            return Err(AudioError::InvalidData("Codebook has no VQ lookup".into()));
        }
        Err(AudioError::InvalidData(
            "Codebook VQ decode not implemented".into(),
        ))
    }

    /// Look up VQ vector for entry index.
    #[must_use]
    #[allow(dead_code, clippy::cast_precision_loss)]
    pub fn lookup(&self, index: usize) -> Option<Vec<f32>> {
        if !self.lookup_type.has_lookup() || index >= self.entries {
            return None;
        }

        match self.lookup_type {
            LookupType::Lattice => {
                // Type 1: compute values from multiplicands
                let mut values = Vec::with_capacity(self.dimensions as usize);
                let mut lookup_offset = index;
                let lookup_values = self.multiplicands.len();

                for _ in 0..self.dimensions {
                    if lookup_values == 0 {
                        break;
                    }
                    let multiplicand = self.multiplicands[lookup_offset % lookup_values];
                    let value = self.minimum_value + (multiplicand as f32) * self.delta_value;
                    values.push(value);
                    lookup_offset /= lookup_values;
                }
                Some(values)
            }
            LookupType::Tessellated => {
                // Type 2: values stored directly
                let start = index * self.dimensions as usize;
                let end = start + self.dimensions as usize;
                if end <= self.multiplicands.len() {
                    let values: Vec<f32> = self.multiplicands[start..end]
                        .iter()
                        .map(|&m| self.minimum_value + (m as f32) * self.delta_value)
                        .collect();
                    Some(values)
                } else {
                    None
                }
            }
            LookupType::None => None,
        }
    }
}

/// Codebook collection for a Vorbis stream.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct CodebookSet {
    /// All codebooks in the stream.
    codebooks: Vec<Codebook>,
}

impl CodebookSet {
    /// Create a new empty codebook set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            codebooks: Vec::new(),
        }
    }

    /// Add a codebook to the set.
    pub fn add(&mut self, codebook: Codebook) {
        self.codebooks.push(codebook);
    }

    /// Get codebook by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Codebook> {
        self.codebooks.get(index)
    }

    /// Get number of codebooks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codebooks.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codebooks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_entry_unused() {
        let entry = CodebookEntry::unused();
        assert!(!entry.used);
        assert!(!entry.is_valid());
    }

    #[test]
    fn test_codebook_entry_new() {
        let entry = CodebookEntry::new(5);
        assert!(entry.used);
        assert_eq!(entry.length, 5);
        assert!(entry.is_valid());
    }

    #[test]
    fn test_codebook_entry_zero_length() {
        let entry = CodebookEntry::new(0);
        assert!(!entry.used);
        assert!(!entry.is_valid());
    }

    #[test]
    fn test_huffman_tree_new() {
        let tree = HuffmanTree::new();
        assert_eq!(tree.max_length(), 0);
        assert_eq!(tree.entry_count(), 0);
    }

    #[test]
    fn test_huffman_tree_build() {
        let lengths = vec![2, 2, 3, 3, 3, 3];
        let tree = HuffmanTree::build(&lengths).expect("should succeed");
        assert_eq!(tree.max_length(), 3);
        assert_eq!(tree.entry_count(), 6);
    }

    #[test]
    fn test_huffman_tree_empty() {
        let tree = HuffmanTree::build(&[]).expect("should succeed");
        assert_eq!(tree.max_length(), 0);
        assert_eq!(tree.entry_count(), 0);
    }

    #[test]
    fn test_lookup_type() {
        assert_eq!(LookupType::from_value(0), Some(LookupType::None));
        assert_eq!(LookupType::from_value(1), Some(LookupType::Lattice));
        assert_eq!(LookupType::from_value(2), Some(LookupType::Tessellated));
        assert_eq!(LookupType::from_value(3), None);
    }

    #[test]
    fn test_lookup_type_has_lookup() {
        assert!(!LookupType::None.has_lookup());
        assert!(LookupType::Lattice.has_lookup());
        assert!(LookupType::Tessellated.has_lookup());
    }

    #[test]
    fn test_codebook_new() {
        let codebook = Codebook::new(0);
        assert_eq!(codebook.id, 0);
        assert_eq!(codebook.entry_count(), 0);
    }

    #[test]
    fn test_codebook_set() {
        let mut set = CodebookSet::new();
        assert!(set.is_empty());

        set.add(Codebook::new(0));
        set.add(Codebook::new(1));

        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        assert!(set.get(0).is_some());
        assert!(set.get(1).is_some());
        assert!(set.get(2).is_none());
    }

    #[test]
    fn test_codebook_lookup_no_vq() {
        let codebook = Codebook::new(0);
        assert!(codebook.lookup(0).is_none());
    }
}
