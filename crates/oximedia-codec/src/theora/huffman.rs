// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Huffman coding for Theora.
//!
//! Implements Huffman encoding and decoding for DCT coefficients
//! as specified in the Theora specification.

use crate::error::{CodecError, CodecResult};
use crate::theora::bitstream::{BitstreamReader, BitstreamWriter};
use crate::theora::tables::{AC_HUFF_TABLES, DC_HUFF_LENGTHS};

/// Maximum Huffman code length.
const MAX_CODE_LENGTH: usize = 32;

/// Huffman tree node.
#[derive(Debug, Clone)]
enum HuffmanNode {
    /// Leaf node with symbol value.
    Leaf(i16),
    /// Internal node with left and right children.
    Internal {
        left: Box<HuffmanNode>,
        right: Box<HuffmanNode>,
    },
}

/// Huffman decoder.
pub struct HuffmanDecoder {
    /// Root of the Huffman tree.
    root: HuffmanNode,
}

impl HuffmanDecoder {
    /// Create a new Huffman decoder from code lengths.
    ///
    /// # Arguments
    ///
    /// * `lengths` - Array of code lengths for each symbol
    pub fn new(lengths: &[u8]) -> CodecResult<Self> {
        let root = Self::build_tree(lengths)?;
        Ok(Self { root })
    }

    /// Build a Huffman tree from code lengths.
    fn build_tree(lengths: &[u8]) -> CodecResult<HuffmanNode> {
        // Count codes of each length
        let mut length_counts = [0u32; MAX_CODE_LENGTH + 1];
        for &len in lengths {
            if (len as usize) <= MAX_CODE_LENGTH {
                length_counts[len as usize] += 1;
            }
        }

        // Generate canonical Huffman codes
        let mut next_code = [0u32; MAX_CODE_LENGTH + 1];
        let mut code = 0u32;
        for i in 1..=MAX_CODE_LENGTH {
            code = (code + length_counts[i - 1]) << 1;
            next_code[i] = code;
        }

        // Build the tree
        let mut root = HuffmanNode::Internal {
            left: Box::new(HuffmanNode::Leaf(-1)),
            right: Box::new(HuffmanNode::Leaf(-1)),
        };

        for (symbol, &len) in lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }

            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            Self::insert_code(&mut root, code, len, symbol as i16)?;
        }

        Ok(root)
    }

    /// Insert a code into the Huffman tree.
    fn insert_code(node: &mut HuffmanNode, code: u32, length: u8, symbol: i16) -> CodecResult<()> {
        if length == 0 {
            *node = HuffmanNode::Leaf(symbol);
            return Ok(());
        }

        match node {
            HuffmanNode::Leaf(_) => {
                *node = HuffmanNode::Internal {
                    left: Box::new(HuffmanNode::Leaf(-1)),
                    right: Box::new(HuffmanNode::Leaf(-1)),
                };
            }
            HuffmanNode::Internal { .. } => {}
        }

        let bit = (code >> (length - 1)) & 1;
        match node {
            HuffmanNode::Internal { left, right } => {
                if bit == 0 {
                    Self::insert_code(left, code, length - 1, symbol)?;
                } else {
                    Self::insert_code(right, code, length - 1, symbol)?;
                }
            }
            HuffmanNode::Leaf(_) => {
                return Err(CodecError::InvalidBitstream(
                    "Invalid Huffman tree structure".to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Decode a single symbol from the bitstream.
    pub fn decode(&self, reader: &mut BitstreamReader) -> CodecResult<i16> {
        let mut node = &self.root;

        loop {
            match node {
                HuffmanNode::Leaf(symbol) => {
                    if *symbol == -1 {
                        return Err(CodecError::InvalidBitstream(
                            "Invalid Huffman code".to_string(),
                        ));
                    }
                    return Ok(*symbol);
                }
                HuffmanNode::Internal { left, right } => {
                    let bit = reader.read_bit()?;
                    node = if bit { right } else { left };
                }
            }
        }
    }
}

/// Huffman encoder.
pub struct HuffmanEncoder {
    /// Code for each symbol.
    codes: Vec<u32>,
    /// Code length for each symbol.
    lengths: Vec<u8>,
}

impl HuffmanEncoder {
    /// Create a new Huffman encoder from code lengths.
    pub fn new(lengths: &[u8]) -> Self {
        let mut codes = vec![0u32; lengths.len()];

        // Generate canonical Huffman codes
        let mut length_counts = [0u32; MAX_CODE_LENGTH + 1];
        for &len in lengths {
            if (len as usize) <= MAX_CODE_LENGTH {
                length_counts[len as usize] += 1;
            }
        }

        let mut next_code = [0u32; MAX_CODE_LENGTH + 1];
        let mut code = 0u32;
        for i in 1..=MAX_CODE_LENGTH {
            code = (code + length_counts[i - 1]) << 1;
            next_code[i] = code;
        }

        for (symbol, &len) in lengths.iter().enumerate() {
            if len > 0 {
                codes[symbol] = next_code[len as usize];
                next_code[len as usize] += 1;
            }
        }

        Self {
            codes,
            lengths: lengths.to_vec(),
        }
    }

    /// Encode a symbol to the bitstream.
    pub fn encode(&self, writer: &mut BitstreamWriter, symbol: i16) -> CodecResult<()> {
        let symbol = symbol as usize;
        if symbol >= self.codes.len() {
            return Err(CodecError::InvalidParameter(format!(
                "Symbol {symbol} out of range"
            )));
        }

        let code = self.codes[symbol];
        let length = self.lengths[symbol];

        if length == 0 {
            return Err(CodecError::InvalidParameter(format!(
                "Symbol {symbol} has no code"
            )));
        }

        writer.write_bits(code, length);
        Ok(())
    }
}

/// Theora DCT token decoder.
pub struct TheoraTokenDecoder {
    /// DC coefficient decoders (one per quality index).
    dc_decoders: Vec<HuffmanDecoder>,
    /// AC coefficient decoders (80 tables).
    ac_decoders: Vec<HuffmanDecoder>,
}

impl TheoraTokenDecoder {
    /// Create a new Theora token decoder.
    pub fn new() -> CodecResult<Self> {
        // Build DC decoders
        let mut dc_decoders = Vec::new();
        for table in &DC_HUFF_LENGTHS {
            dc_decoders.push(HuffmanDecoder::new(table)?);
        }

        // Build AC decoders
        let mut ac_decoders = Vec::new();
        for table in &AC_HUFF_TABLES {
            ac_decoders.push(HuffmanDecoder::new(table)?);
        }

        Ok(Self {
            dc_decoders,
            ac_decoders,
        })
    }

    /// Decode a DC coefficient token.
    pub fn decode_dc(&self, reader: &mut BitstreamReader, table: usize) -> CodecResult<i16> {
        if table >= self.dc_decoders.len() {
            return Err(CodecError::InvalidParameter(format!(
                "DC table index {table} out of range"
            )));
        }
        self.dc_decoders[table].decode(reader)
    }

    /// Decode an AC coefficient token.
    pub fn decode_ac(&self, reader: &mut BitstreamReader, table: usize) -> CodecResult<i16> {
        if table >= self.ac_decoders.len() {
            return Err(CodecError::InvalidParameter(format!(
                "AC table index {table} out of range"
            )));
        }
        self.ac_decoders[table].decode(reader)
    }
}

impl Default for TheoraTokenDecoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default token decoder")
    }
}

/// Theora DCT token encoder.
pub struct TheoraTokenEncoder {
    /// DC coefficient encoders.
    dc_encoders: Vec<HuffmanEncoder>,
    /// AC coefficient encoders.
    ac_encoders: Vec<HuffmanEncoder>,
}

impl TheoraTokenEncoder {
    /// Create a new Theora token encoder.
    #[must_use]
    pub fn new() -> Self {
        // Build DC encoders
        let mut dc_encoders = Vec::new();
        for table in &DC_HUFF_LENGTHS {
            dc_encoders.push(HuffmanEncoder::new(table));
        }

        // Build AC encoders
        let mut ac_encoders = Vec::new();
        for table in &AC_HUFF_TABLES {
            ac_encoders.push(HuffmanEncoder::new(table));
        }

        Self {
            dc_encoders,
            ac_encoders,
        }
    }

    /// Encode a DC coefficient token.
    pub fn encode_dc(
        &self,
        writer: &mut BitstreamWriter,
        table: usize,
        token: i16,
    ) -> CodecResult<()> {
        if table >= self.dc_encoders.len() {
            return Err(CodecError::InvalidParameter(format!(
                "DC table index {table} out of range"
            )));
        }
        self.dc_encoders[table].encode(writer, token)
    }

    /// Encode an AC coefficient token.
    pub fn encode_ac(
        &self,
        writer: &mut BitstreamWriter,
        table: usize,
        token: i16,
    ) -> CodecResult<()> {
        if table >= self.ac_encoders.len() {
            return Err(CodecError::InvalidParameter(format!(
                "AC table index {table} out of range"
            )));
        }
        self.ac_encoders[table].encode(writer, token)
    }
}

impl Default for TheoraTokenEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode a run-length encoded coefficient token.
///
/// Returns (run, value) where run is the number of zeros before the value.
pub fn decode_token(token: i16) -> (usize, i16) {
    if token == 0 {
        // End of block
        (0, 0)
    } else if token < 16 {
        // Single coefficient
        (0, token)
    } else {
        // Run-length encoded
        let run = ((token >> 4) & 0x0F) as usize;
        let value = token & 0x0F;
        (run, value)
    }
}

/// Encode a coefficient with run-length encoding.
pub fn encode_token(run: usize, value: i16) -> i16 {
    if run == 0 && value == 0 {
        // End of block
        0
    } else if run == 0 {
        // Single coefficient
        value
    } else {
        // Run-length encoded
        ((run as i16) << 4) | (value & 0x0F)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_coding() {
        let lengths = [3u8, 3, 3, 3, 4, 4, 4, 4];
        let encoder = HuffmanEncoder::new(&lengths);
        let decoder = HuffmanDecoder::new(&lengths).expect("should succeed");

        let mut writer = BitstreamWriter::new();
        encoder
            .encode(&mut writer, 0)
            .expect("encode should succeed");
        encoder
            .encode(&mut writer, 5)
            .expect("encode should succeed");
        writer.byte_align();

        let data = writer.into_vec();
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(
            decoder.decode(&mut reader).expect("decode should succeed"),
            0
        );
        assert_eq!(
            decoder.decode(&mut reader).expect("decode should succeed"),
            5
        );
    }

    #[test]
    fn test_token_encoding() {
        assert_eq!(encode_token(0, 5), 5);
        assert_eq!(encode_token(3, 7), 0x37);
        assert_eq!(encode_token(0, 0), 0);

        assert_eq!(decode_token(5), (0, 5));
        assert_eq!(decode_token(0x37), (3, 7));
        assert_eq!(decode_token(0), (0, 0));
    }

    #[test]
    fn test_theora_token_decoder() {
        let decoder = TheoraTokenDecoder::new().expect("should succeed");
        assert_eq!(decoder.dc_decoders.len(), 16);
        assert_eq!(decoder.ac_decoders.len(), 80);
    }
}
