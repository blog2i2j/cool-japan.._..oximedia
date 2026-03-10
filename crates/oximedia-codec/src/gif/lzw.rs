//! LZW compression and decompression for GIF format.
//!
//! Implements the Lempel-Ziv-Welch (LZW) algorithm with variable-length codes
//! as specified in the GIF89a standard.

use crate::error::{CodecError, CodecResult};
use std::collections::HashMap;

/// Maximum code size in bits (12 bits for GIF).
const MAX_CODE_SIZE: u8 = 12;

/// Maximum number of codes in the table.
const MAX_CODES: usize = 1 << MAX_CODE_SIZE;

/// LZW decompressor for GIF data.
///
/// Decompresses data encoded with the LZW algorithm as specified
/// in the GIF89a standard.
pub struct LzwDecoder {
    /// Minimum code size (initial bits per code).
    min_code_size: u8,
    /// Current code size in bits.
    code_size: u8,
    /// Clear code value.
    clear_code: u16,
    /// End of information code value.
    eoi_code: u16,
    /// Next available code.
    next_code: u16,
    /// Code table storing decompressed sequences.
    table: Vec<Vec<u8>>,
    /// Previous code.
    prev_code: Option<u16>,
    /// Output buffer for decompressed data.
    output: Vec<u8>,
}

impl LzwDecoder {
    /// Create a new LZW decoder.
    ///
    /// # Arguments
    ///
    /// * `min_code_size` - Minimum code size in bits (2-8 for GIF)
    ///
    /// # Errors
    ///
    /// Returns error if `min_code_size` is invalid.
    pub fn new(min_code_size: u8) -> CodecResult<Self> {
        if !(2..=8).contains(&min_code_size) {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid LZW min code size: {}",
                min_code_size
            )));
        }

        let clear_code = 1 << min_code_size;
        let eoi_code = clear_code + 1;
        let code_size = min_code_size + 1;

        Ok(Self {
            min_code_size,
            code_size,
            clear_code,
            eoi_code,
            next_code: eoi_code + 1,
            table: Vec::with_capacity(MAX_CODES),
            prev_code: None,
            output: Vec::new(),
        })
    }

    /// Initialize the code table with single-byte entries.
    fn init_table(&mut self) {
        self.table.clear();
        let table_size = 1 << self.min_code_size;

        // Add all single-byte codes
        for i in 0..table_size {
            self.table.push(vec![i as u8]);
        }

        // Reserve space for clear code and EOI code
        self.table.push(Vec::new()); // Clear code
        self.table.push(Vec::new()); // EOI code

        self.next_code = self.eoi_code + 1;
        self.code_size = self.min_code_size + 1;
        self.prev_code = None;
    }

    /// Decompress LZW-encoded data.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed data bytes
    /// * `output_size` - Expected output size (for validation)
    ///
    /// # Errors
    ///
    /// Returns error if decompression fails or data is invalid.
    pub fn decompress(&mut self, data: &[u8], output_size: usize) -> CodecResult<Vec<u8>> {
        self.output.clear();
        self.output.reserve(output_size);
        self.init_table();

        let mut bit_reader = BitReader::new(data);

        loop {
            // Read next code
            let code = match bit_reader.read_bits(self.code_size) {
                Some(c) => c,
                None => break, // End of data
            };

            // Handle special codes
            if code == self.clear_code {
                self.init_table();
                continue;
            }

            if code == self.eoi_code {
                break;
            }

            // Process the code
            if let Some(sequence) = self.get_sequence(code)? {
                // Output the sequence
                self.output.extend_from_slice(&sequence);

                // Add new entry to the table
                if let Some(prev) = self.prev_code {
                    if self.next_code < MAX_CODES as u16 {
                        let mut new_entry = self.get_sequence(prev)?.ok_or_else(|| {
                            CodecError::InvalidData(format!("LZW prev_code {} not in table", prev))
                        })?;
                        new_entry.push(sequence[0]);
                        self.table.push(new_entry);
                        self.next_code += 1;

                        // Increase code size if needed
                        if self.next_code >= (1 << self.code_size) && self.code_size < MAX_CODE_SIZE
                        {
                            self.code_size += 1;
                        }
                    }
                }

                self.prev_code = Some(code);
            } else {
                return Err(CodecError::InvalidData(format!(
                    "Invalid LZW code: {}",
                    code
                )));
            }
        }

        Ok(self.output.clone())
    }

    /// Get the sequence for a given code.
    fn get_sequence(&self, code: u16) -> CodecResult<Option<Vec<u8>>> {
        let code_idx = code as usize;

        if code_idx < self.table.len() {
            Ok(Some(self.table[code_idx].clone()))
        } else if code == self.next_code {
            // Special case: code not in table yet (KwKwK pattern)
            if let Some(prev) = self.prev_code {
                let mut sequence = self.table[prev as usize].clone();
                sequence.push(sequence[0]);
                Ok(Some(sequence))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}

/// LZW compressor for GIF data.
///
/// Compresses data using the LZW algorithm as specified
/// in the GIF89a standard.
pub struct LzwEncoder {
    /// Minimum code size (initial bits per code).
    min_code_size: u8,
    /// Current code size in bits.
    code_size: u8,
    /// Clear code value.
    clear_code: u16,
    /// End of information code value.
    eoi_code: u16,
    /// Next available code.
    next_code: u16,
    /// Code table mapping sequences to codes.
    table: HashMap<Vec<u8>, u16>,
    /// Output buffer for compressed data.
    output: Vec<u8>,
}

impl LzwEncoder {
    /// Create a new LZW encoder.
    ///
    /// # Arguments
    ///
    /// * `min_code_size` - Minimum code size in bits (2-8 for GIF)
    ///
    /// # Errors
    ///
    /// Returns error if `min_code_size` is invalid.
    pub fn new(min_code_size: u8) -> CodecResult<Self> {
        if !(2..=8).contains(&min_code_size) {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid LZW min code size: {}",
                min_code_size
            )));
        }

        let clear_code = 1 << min_code_size;
        let eoi_code = clear_code + 1;
        let code_size = min_code_size + 1;

        Ok(Self {
            min_code_size,
            code_size,
            clear_code,
            eoi_code,
            next_code: eoi_code + 1,
            table: HashMap::new(),
            output: Vec::new(),
        })
    }

    /// Initialize the code table with single-byte entries.
    fn init_table(&mut self) {
        self.table.clear();
        let table_size = 1 << self.min_code_size;

        // Add all single-byte codes
        for i in 0..table_size {
            self.table.insert(vec![i as u8], i);
        }

        self.next_code = self.eoi_code + 1;
        self.code_size = self.min_code_size + 1;
    }

    /// Compress data using LZW algorithm.
    ///
    /// # Arguments
    ///
    /// * `data` - Data to compress
    ///
    /// # Errors
    ///
    /// Returns error if compression fails.
    pub fn compress(&mut self, data: &[u8]) -> CodecResult<Vec<u8>> {
        self.output.clear();
        self.init_table();

        let mut bit_writer = BitWriter::new();

        // Write initial clear code
        bit_writer.write_bits(self.clear_code.into(), self.code_size);

        let mut current_sequence = Vec::new();

        for &byte in data {
            let mut new_sequence = current_sequence.clone();
            new_sequence.push(byte);

            if self.table.contains_key(&new_sequence) {
                // Sequence exists in table, continue building
                current_sequence = new_sequence;
            } else {
                // Output code for current sequence
                if let Some(&code) = self.table.get(&current_sequence) {
                    bit_writer.write_bits(code.into(), self.code_size);
                }

                // Add new sequence to table
                if self.next_code < MAX_CODES as u16 {
                    self.table.insert(new_sequence.clone(), self.next_code);
                    self.next_code += 1;

                    // Increase code size if needed
                    if self.next_code > (1 << self.code_size) && self.code_size < MAX_CODE_SIZE {
                        self.code_size += 1;
                    }
                }

                // Reset to single byte
                current_sequence = vec![byte];

                // Check if we need to reset the table
                if self.next_code >= MAX_CODES as u16 {
                    bit_writer.write_bits(self.clear_code.into(), self.code_size);
                    self.init_table();
                }
            }
        }

        // Output final sequence
        if !current_sequence.is_empty() {
            if let Some(&code) = self.table.get(&current_sequence) {
                bit_writer.write_bits(code.into(), self.code_size);
            }
        }

        // Write EOI code
        bit_writer.write_bits(self.eoi_code.into(), self.code_size);

        // Flush remaining bits
        self.output = bit_writer.finish();
        Ok(self.output.clone())
    }
}

/// Bit reader for reading variable-length codes from byte stream.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader.
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read `n` bits from the stream.
    fn read_bits(&mut self, n: u8) -> Option<u16> {
        if n > 16 {
            return None;
        }

        let mut result: u16 = 0;
        let mut bits_read = 0;

        while bits_read < n {
            if self.byte_pos >= self.data.len() {
                return None;
            }

            let byte = self.data[self.byte_pos];
            let bits_available = 8 - self.bit_pos;
            let bits_needed = n - bits_read;
            let bits_to_read = bits_available.min(bits_needed);

            // Extract bits from current byte
            let mask = ((1u32 << bits_to_read) - 1) as u8;
            let bits = (byte >> self.bit_pos) & mask;
            result |= u16::from(bits) << bits_read;

            bits_read += bits_to_read;
            self.bit_pos += bits_to_read;

            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        Some(result)
    }
}

/// Bit writer for writing variable-length codes to byte stream.
struct BitWriter {
    data: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
}

impl BitWriter {
    /// Create a new bit writer.
    fn new() -> Self {
        Self {
            data: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    /// Write `n` bits to the stream.
    fn write_bits(&mut self, value: u32, n: u8) {
        let mut value = value;
        let mut bits_written = 0;

        while bits_written < n {
            let bits_available = 8 - self.bit_pos;
            let bits_remaining = n - bits_written;
            let bits_to_write = bits_available.min(bits_remaining);

            // Extract bits to write
            let mask = (1 << bits_to_write) - 1;
            let bits = (value & mask) as u8;

            // Write bits to current byte
            self.current_byte |= bits << self.bit_pos;

            value >>= bits_to_write;
            bits_written += bits_to_write;
            self.bit_pos += bits_to_write;

            if self.bit_pos >= 8 {
                self.data.push(self.current_byte);
                self.current_byte = 0;
                self.bit_pos = 0;
            }
        }
    }

    /// Finish writing and return the data.
    fn finish(mut self) -> Vec<u8> {
        // Flush remaining bits
        if self.bit_pos > 0 {
            self.data.push(self.current_byte);
        }
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lzw_roundtrip() {
        let min_code_size = 8;
        let mut encoder = LzwEncoder::new(min_code_size).expect("should succeed");
        let mut decoder = LzwDecoder::new(min_code_size).expect("should succeed");

        let original = b"TOBEORNOTTOBEORTOBEORNOT";
        let compressed = encoder.compress(original).expect("should succeed");
        let decompressed = decoder
            .decompress(&compressed, original.len())
            .expect("should succeed");

        assert_eq!(original, decompressed.as_slice());
    }

    #[test]
    fn test_lzw_simple() {
        let min_code_size = 2;
        let mut encoder = LzwEncoder::new(min_code_size).expect("should succeed");
        let mut decoder = LzwDecoder::new(min_code_size).expect("should succeed");

        let original = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let compressed = encoder.compress(&original).expect("should succeed");
        let decompressed = decoder
            .decompress(&compressed, original.len())
            .expect("should succeed");

        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_lzw_repeated_pattern() {
        let min_code_size = 8;
        let mut encoder = LzwEncoder::new(min_code_size).expect("should succeed");
        let mut decoder = LzwDecoder::new(min_code_size).expect("should succeed");

        let original = vec![1; 1000];
        let compressed = encoder.compress(&original).expect("should succeed");
        let decompressed = decoder
            .decompress(&compressed, original.len())
            .expect("should succeed");

        assert_eq!(original, decompressed);
        assert!(compressed.len() < original.len());
    }

    #[test]
    fn test_bit_reader_writer() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.write_bits(0b110, 3);
        writer.write_bits(0b1111, 4);
        let data = writer.finish();

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(3), Some(0b101));
        assert_eq!(reader.read_bits(3), Some(0b110));
        assert_eq!(reader.read_bits(4), Some(0b1111));
    }

    #[test]
    fn test_invalid_min_code_size() {
        assert!(LzwEncoder::new(1).is_err());
        assert!(LzwEncoder::new(9).is_err());
        assert!(LzwDecoder::new(1).is_err());
        assert!(LzwDecoder::new(9).is_err());
    }
}
