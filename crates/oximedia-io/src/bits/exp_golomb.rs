//! Exp-Golomb coding implementation.
//!
//! Exponential-Golomb (Exp-Golomb) codes are variable-length codes used
//! in H.264/AVC and other video coding standards for encoding integers.
//!
//! # Encoding Format
//!
//! An Exp-Golomb code consists of:
//! 1. A prefix of `M` zero bits
//! 2. A separator `1` bit
//! 3. A suffix of `M` information bits
//!
//! The value is calculated as: `2^M + suffix - 1`
//!
//! # Examples
//!
//! | Value | Code     | Binary  |
//! |-------|----------|---------|
//! | 0     | 1        | 1       |
//! | 1     | 010      | 010     |
//! | 2     | 011      | 011     |
//! | 3     | 00100    | 00100   |
//! | 4     | 00101    | 00101   |
//!
//! # Signed Values
//!
//! Signed Exp-Golomb (se(v)) maps unsigned values to signed:
//! - 0 -> 0
//! - 1 -> 1
//! - 2 -> -1
//! - 3 -> 2
//! - 4 -> -2

use super::BitReader;
use oximedia_core::{OxiError, OxiResult};

impl BitReader<'_> {
    /// Reads an unsigned Exp-Golomb coded integer (ue(v)).
    ///
    /// This is used extensively in H.264 for encoding syntax elements.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are not enough bits.
    /// Returns [`OxiError::InvalidData`] if the code is malformed or too long.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // ue(0) = 1 (single bit)
    /// let data = [0b10000000];
    /// let mut reader = BitReader::new(&data);
    /// assert_eq!(reader.read_exp_golomb()?, 0);
    ///
    /// // ue(1) = 010
    /// let data = [0b01000000];
    /// let mut reader = BitReader::new(&data);
    /// assert_eq!(reader.read_exp_golomb()?, 1);
    ///
    /// // ue(2) = 011
    /// let data = [0b01100000];
    /// let mut reader = BitReader::new(&data);
    /// assert_eq!(reader.read_exp_golomb()?, 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_exp_golomb(&mut self) -> OxiResult<u64> {
        // Count leading zeros
        let mut leading_zeros: u8 = 0;
        while self.read_bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 63 {
                return Err(OxiError::InvalidData(
                    "Exp-Golomb code too long (> 63 leading zeros)".to_string(),
                ));
            }
        }

        if leading_zeros == 0 {
            return Ok(0);
        }

        // Read the suffix bits
        let suffix = self.read_bits(leading_zeros)?;

        // Calculate value: 2^M + suffix - 1
        Ok((1u64 << leading_zeros) - 1 + suffix)
    }

    /// Reads a signed Exp-Golomb coded integer (se(v)).
    ///
    /// Maps unsigned Exp-Golomb values to signed:
    /// - 0 -> 0
    /// - 1 -> 1
    /// - 2 -> -1
    /// - 3 -> 2
    /// - 4 -> -2
    /// - etc.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are not enough bits.
    /// Returns [`OxiError::InvalidData`] if the code is malformed.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::bits::BitReader;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // se(0) = 1 -> value 0
    /// let data = [0b10000000];
    /// let mut reader = BitReader::new(&data);
    /// assert_eq!(reader.read_signed_exp_golomb()?, 0);
    ///
    /// // se(+1) = 010 -> value 1
    /// let data = [0b01000000];
    /// let mut reader = BitReader::new(&data);
    /// assert_eq!(reader.read_signed_exp_golomb()?, 1);
    ///
    /// // se(-1) = 011 -> value 2
    /// let data = [0b01100000];
    /// let mut reader = BitReader::new(&data);
    /// assert_eq!(reader.read_signed_exp_golomb()?, -1);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::cast_possible_wrap)]
    pub fn read_signed_exp_golomb(&mut self) -> OxiResult<i64> {
        let ue = self.read_exp_golomb()?;

        // Map: ue -> se
        // 0 -> 0, 1 -> 1, 2 -> -1, 3 -> 2, 4 -> -2, ...
        // Formula: if odd, positive (ue+1)/2; if even, negative -(ue/2)
        let abs_value = ue.div_ceil(2) as i64;
        if ue & 1 == 0 {
            Ok(-abs_value)
        } else {
            Ok(abs_value)
        }
    }

    /// Reads an unsigned Exp-Golomb coded integer, alias for `read_exp_golomb`.
    ///
    /// This follows H.264 naming convention (ue(v)).
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are not enough bits.
    /// Returns [`OxiError::InvalidData`] if the code is malformed.
    #[inline]
    pub fn read_ue(&mut self) -> OxiResult<u64> {
        self.read_exp_golomb()
    }

    /// Reads a signed Exp-Golomb coded integer, alias for `read_signed_exp_golomb`.
    ///
    /// This follows H.264 naming convention (se(v)).
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::UnexpectedEof`] if there are not enough bits.
    /// Returns [`OxiError::InvalidData`] if the code is malformed.
    #[inline]
    pub fn read_se(&mut self) -> OxiResult<i64> {
        self.read_signed_exp_golomb()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_exp_golomb_zero() {
        // ue(0) = 1
        let data = [0b10000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            0
        );
        assert_eq!(reader.bits_read(), 1);
    }

    #[test]
    fn test_read_exp_golomb_one() {
        // ue(1) = 010
        let data = [0b01000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            1
        );
        assert_eq!(reader.bits_read(), 3);
    }

    #[test]
    fn test_read_exp_golomb_two() {
        // ue(2) = 011
        let data = [0b01100000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            2
        );
        assert_eq!(reader.bits_read(), 3);
    }

    #[test]
    fn test_read_exp_golomb_three() {
        // ue(3) = 00100
        let data = [0b00100000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            3
        );
        assert_eq!(reader.bits_read(), 5);
    }

    #[test]
    fn test_read_exp_golomb_four() {
        // ue(4) = 00101
        let data = [0b00101000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            4
        );
        assert_eq!(reader.bits_read(), 5);
    }

    #[test]
    fn test_read_exp_golomb_five() {
        // ue(5) = 00110
        let data = [0b00110000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            5
        );
        assert_eq!(reader.bits_read(), 5);
    }

    #[test]
    fn test_read_exp_golomb_six() {
        // ue(6) = 00111
        let data = [0b00111000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            6
        );
        assert_eq!(reader.bits_read(), 5);
    }

    #[test]
    fn test_read_exp_golomb_seven() {
        // ue(7) = 0001000
        let data = [0b00010000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            7
        );
        assert_eq!(reader.bits_read(), 7);
    }

    #[test]
    fn test_read_exp_golomb_large() {
        // ue(14) = 0001111
        let data = [0b00011110];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            14
        );
    }

    #[test]
    fn test_read_signed_exp_golomb_zero() {
        // se(0) = ue(0) = 1
        let data = [0b10000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_signed_exp_golomb()
                .expect("read_signed_exp_golomb should succeed"),
            0
        );
    }

    #[test]
    fn test_read_signed_exp_golomb_positive_one() {
        // se(+1) = ue(1) = 010
        let data = [0b01000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_signed_exp_golomb()
                .expect("read_signed_exp_golomb should succeed"),
            1
        );
    }

    #[test]
    fn test_read_signed_exp_golomb_negative_one() {
        // se(-1) = ue(2) = 011
        let data = [0b01100000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_signed_exp_golomb()
                .expect("read_signed_exp_golomb should succeed"),
            -1
        );
    }

    #[test]
    fn test_read_signed_exp_golomb_positive_two() {
        // se(+2) = ue(3) = 00100
        let data = [0b00100000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_signed_exp_golomb()
                .expect("read_signed_exp_golomb should succeed"),
            2
        );
    }

    #[test]
    fn test_read_signed_exp_golomb_negative_two() {
        // se(-2) = ue(4) = 00101
        let data = [0b00101000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_signed_exp_golomb()
                .expect("read_signed_exp_golomb should succeed"),
            -2
        );
    }

    #[test]
    fn test_read_signed_exp_golomb_sequence() {
        // Test the mapping: 0->0, 1->1, 2->-1, 3->2, 4->-2, 5->3, 6->-3
        let test_cases: [(u64, i64); 7] =
            [(0, 0), (1, 1), (2, -1), (3, 2), (4, -2), (5, 3), (6, -3)];

        for (ue_val, expected_se) in test_cases {
            let abs_value = ((ue_val + 1) / 2) as i64;
            let se_val = if ue_val & 1 == 0 {
                -abs_value
            } else {
                abs_value
            };
            assert_eq!(
                se_val, expected_se,
                "ue({ue_val}) should map to se({expected_se})"
            );
        }
    }

    #[test]
    fn test_read_multiple_exp_golomb() {
        // Two values: ue(0)=1 and ue(1)=010 packed together
        let data = [0b10100000];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            0
        );
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            1
        );
    }

    #[test]
    fn test_read_ue_alias() {
        let data = [0b01000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_ue().expect("read_ue should succeed"), 1);
    }

    #[test]
    fn test_read_se_alias() {
        let data = [0b01100000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_se().expect("read_se should succeed"), -1);
    }

    #[test]
    fn test_exp_golomb_eof() {
        // Not enough bits
        let data = [0b00000000];
        let mut reader = BitReader::new(&data);
        let result = reader.read_exp_golomb();
        assert!(result.is_err());
    }

    // Additional comprehensive tests

    #[test]
    fn test_exp_golomb_boundary_values() {
        // Test values at power-of-2 boundaries
        // ue(14) = 0001111 (3 leading zeros, suffix=111)
        let data = [0b00011110];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            14
        );

        // ue(30) = 00011111 (3 leading zeros, suffix = 111)
        let data = [0b00011111];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            14
        );
    }

    #[test]
    fn test_exp_golomb_consecutive_zeros() {
        // Multiple ue(0) values in a row
        let data = [0b11110000]; // Four ue(0) values
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            0
        );
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            0
        );
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            0
        );
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            0
        );
    }

    #[test]
    fn test_signed_exp_golomb_range() {
        // Test the full mapping for small values
        let test_cases = [
            (0b10000000, 0),  // ue(0) -> se(0)
            (0b01000000, 1),  // ue(1) -> se(1)
            (0b01100000, -1), // ue(2) -> se(-1)
            (0b00100000, 2),  // ue(3) -> se(2)
            (0b00101000, -2), // ue(4) -> se(-2)
            (0b00110000, 3),  // ue(5) -> se(3)
            (0b00111000, -3), // ue(6) -> se(-3)
        ];

        for (data_byte, expected) in test_cases {
            let data = [data_byte];
            let mut reader = BitReader::new(&data);
            assert_eq!(
                reader
                    .read_signed_exp_golomb()
                    .expect("read_signed_exp_golomb should succeed"),
                expected
            );
        }
    }

    #[test]
    fn test_signed_exp_golomb_large_values() {
        // Test larger signed values
        // se(5) = ue(9) = 0001010 (3 leading zeros, suffix=010)
        // Binary: 0001010
        let data = [0b00010100];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_signed_exp_golomb()
                .expect("read_signed_exp_golomb should succeed"),
            5
        );

        // se(-5) = ue(10) = 0001011 (3 leading zeros, suffix=011)
        // Binary: 0001011
        let data = [0b00010110];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_signed_exp_golomb()
                .expect("read_signed_exp_golomb should succeed"),
            -5
        );
    }

    #[test]
    fn test_exp_golomb_mixed_with_other_reads() {
        // Test exp-golomb mixed with regular bit reads
        let data = [0b11100000]; // flag(1), flag(1), ue(0)=1, ...
        let mut reader = BitReader::new(&data);

        assert!(reader.read_flag().expect("read_flag should succeed"));
        assert!(reader.read_flag().expect("read_flag should succeed"));
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            0
        );
    }

    #[test]
    fn test_exp_golomb_too_many_leading_zeros() {
        // Test error handling for too many leading zeros (>63)
        let data = [0x00; 10]; // 80 zero bits
        let mut reader = BitReader::new(&data);
        let result = reader.read_exp_golomb();
        assert!(result.is_err());
    }

    #[test]
    fn test_exp_golomb_insufficient_suffix_bits() {
        // Leading zeros indicate we need more bits than available
        let data = [0b00000001]; // 7 leading zeros, but only 1 bit left
        let mut reader = BitReader::new(&data);
        let result = reader.read_exp_golomb();
        assert!(result.is_err());
    }

    #[test]
    fn test_exp_golomb_arithmetic() {
        // Verify the calculation: 2^M + suffix - 1
        // For ue(10): M=3, suffix=3, value = 2^3 + 3 - 1 = 8 + 3 - 1 = 10
        // Binary: 0001011
        let data = [0b00010110];
        let mut reader = BitReader::new(&data);
        assert_eq!(
            reader
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            10
        );
    }

    #[test]
    fn test_signed_zero_mapping() {
        // Ensure se(0) maps correctly from ue(0)
        let data = [0b10000000];
        let mut reader = BitReader::new(&data);
        let value = reader.read_se().expect("read_se should succeed");
        assert_eq!(value, 0);
    }

    #[test]
    fn test_alternating_signed_pattern() {
        // Test that signed values alternate positive/negative correctly
        // Pack: se(1)=ue(1)=010, se(-1)=ue(2)=011, se(2)=ue(3)=00100
        let data = [
            0b01001100, // 010 011 00...
            0b10000000, // ...100
        ];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_se().expect("read_se should succeed"), 1);
        assert_eq!(reader.read_se().expect("read_se should succeed"), -1);
        assert_eq!(reader.read_se().expect("read_se should succeed"), 2);
    }

    #[test]
    fn test_ue_se_alias_consistency() {
        // Ensure ue/se aliases work identically to full names
        let data = [0b01000000, 0b01100000];
        let mut reader = BitReader::new(&data);

        let ue_val = reader.read_ue().expect("read_ue should succeed");
        let se_val = reader.read_se().expect("read_se should succeed");

        let mut reader2 = BitReader::new(&data);
        assert_eq!(
            reader2
                .read_exp_golomb()
                .expect("read_exp_golomb should succeed"),
            ue_val
        );
        assert_eq!(
            reader2
                .read_signed_exp_golomb()
                .expect("read_signed_exp_golomb should succeed"),
            se_val
        );
    }
}
