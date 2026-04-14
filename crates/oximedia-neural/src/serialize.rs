//! Model weight serialization and deserialization.
//!
//! Saves and loads neural network weights in a simple custom binary format:
//!
//! ```text
//! Header: b"OXNN" (4 bytes magic)
//! Version: u8 (1 byte, currently 1)
//! Num layers: u32 LE
//! For each layer:
//!   Num weights: u32 LE
//!   Weights: f32 LE × num_weights
//! ```
//!
//! No external dependencies — pure Rust byte manipulation.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::serialize::ModelSerializer;
//!
//! let weights = vec![
//!     vec![0.1_f32, 0.2, 0.3],
//!     vec![-0.5_f32, 1.0],
//! ];
//! let bytes = ModelSerializer::to_bytes(&weights);
//! let recovered = ModelSerializer::from_bytes(&bytes).unwrap();
//! assert_eq!(weights, recovered);
//! ```

use crate::error::NeuralError;

/// Magic bytes at the start of every serialized model file.
const MAGIC: &[u8; 4] = b"OXNN";
/// Current format version.
const VERSION: u8 = 1;

/// Serializer / deserializer for model weight vectors.
pub struct ModelSerializer;

impl ModelSerializer {
    /// Serialize a list of weight tensors to bytes.
    ///
    /// Each element in `weights` represents one layer's weight buffer.
    /// Empty layers are serialised as zero-length weight vectors.
    #[must_use]
    pub fn to_bytes(weights: &[Vec<f32>]) -> Vec<u8> {
        // Compute buffer capacity
        let total_floats: usize = weights.iter().map(|w| w.len()).sum();
        let capacity = 4 + 1 + 4 + weights.len() * 4 + total_floats * 4;
        let mut buf = Vec::with_capacity(capacity);

        // Magic
        buf.extend_from_slice(MAGIC);
        // Version
        buf.push(VERSION);
        // Num layers
        let n_layers = weights.len() as u32;
        buf.extend_from_slice(&n_layers.to_le_bytes());

        for layer in weights {
            // Num weights in this layer
            let n = layer.len() as u32;
            buf.extend_from_slice(&n.to_le_bytes());
            // Weight values
            for &w in layer {
                buf.extend_from_slice(&w.to_le_bytes());
            }
        }

        buf
    }

    /// Deserialize bytes into a list of weight tensors.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::InvalidShape`] if the byte slice is malformed,
    /// has an incorrect magic header, or an unsupported version.
    pub fn from_bytes(bytes: &[u8]) -> Result<Vec<Vec<f32>>, NeuralError> {
        let mut pos = 0;

        // Magic
        if bytes.len() < 4 {
            return Err(NeuralError::InvalidShape(
                "ModelSerializer: too short for magic".to_string(),
            ));
        }
        if &bytes[0..4] != MAGIC {
            return Err(NeuralError::InvalidShape(format!(
                "ModelSerializer: invalid magic {:?}",
                &bytes[0..4]
            )));
        }
        pos += 4;

        // Version
        if bytes.len() <= pos {
            return Err(NeuralError::InvalidShape(
                "ModelSerializer: truncated at version".to_string(),
            ));
        }
        let version = bytes[pos];
        pos += 1;
        if version != VERSION {
            return Err(NeuralError::InvalidShape(format!(
                "ModelSerializer: unsupported version {version}"
            )));
        }

        // Num layers
        let n_layers = read_u32(bytes, &mut pos)? as usize;

        let mut weights: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let n_weights = read_u32(bytes, &mut pos).map_err(|_| {
                NeuralError::InvalidShape(format!(
                    "ModelSerializer: truncated reading layer {layer_idx} count"
                ))
            })? as usize;

            let mut layer = Vec::with_capacity(n_weights);
            for w_idx in 0..n_weights {
                let w = read_f32(bytes, &mut pos).map_err(|_| {
                    NeuralError::InvalidShape(format!(
                        "ModelSerializer: truncated at layer {layer_idx} weight {w_idx}"
                    ))
                })?;
                layer.push(w);
            }
            weights.push(layer);
        }

        Ok(weights)
    }

    /// Serialize weights to a byte vector and return the length in bytes.
    ///
    /// Convenience wrapper around [`Self::to_bytes`].
    #[must_use]
    pub fn serialized_len(weights: &[Vec<f32>]) -> usize {
        let total_floats: usize = weights.iter().map(|w| w.len()).sum();
        4 + 1 + 4 + weights.len() * 4 + total_floats * 4
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, NeuralError> {
    let end = *pos + 4;
    if end > bytes.len() {
        return Err(NeuralError::InvalidShape(
            "ModelSerializer: unexpected EOF reading u32".to_string(),
        ));
    }
    let arr: [u8; 4] = bytes[*pos..end]
        .try_into()
        .map_err(|_| NeuralError::InvalidShape("slice conversion".to_string()))?;
    *pos = end;
    Ok(u32::from_le_bytes(arr))
}

fn read_f32(bytes: &[u8], pos: &mut usize) -> Result<f32, NeuralError> {
    let end = *pos + 4;
    if end > bytes.len() {
        return Err(NeuralError::InvalidShape(
            "ModelSerializer: unexpected EOF reading f32".to_string(),
        ));
    }
    let arr: [u8; 4] = bytes[*pos..end]
        .try_into()
        .map_err(|_| NeuralError::InvalidShape("slice conversion".to_string()))?;
    *pos = end;
    Ok(f32::from_le_bytes(arr))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(weights: &[Vec<f32>]) {
        let bytes = ModelSerializer::to_bytes(weights);
        let recovered = ModelSerializer::from_bytes(&bytes)
            .expect("deserialization should succeed for valid bytes");
        assert_eq!(weights.len(), recovered.len(), "layer count mismatch");
        for (i, (orig, rec)) in weights.iter().zip(recovered.iter()).enumerate() {
            assert_eq!(orig.len(), rec.len(), "layer {i} weight count mismatch");
            for (j, (&o, &r)) in orig.iter().zip(rec.iter()).enumerate() {
                assert!((o - r).abs() < 1e-7, "layer {i} weight {j}: {o} != {r}");
            }
        }
    }

    #[test]
    fn test_empty_weights() {
        round_trip(&[]);
    }

    #[test]
    fn test_single_layer() {
        round_trip(&[vec![0.1, -0.2, 0.3, f32::MAX, f32::MIN_POSITIVE]]);
    }

    #[test]
    fn test_multiple_layers() {
        let weights = vec![vec![1.0, 2.0, 3.0], vec![-1.0, 0.5], vec![0.0; 100]];
        round_trip(&weights);
    }

    #[test]
    fn test_empty_layer() {
        round_trip(&[vec![], vec![1.0, 2.0], vec![]]);
    }

    #[test]
    fn test_bad_magic() {
        let bytes = b"BAAD\x01\x00\x00\x00\x00".to_vec();
        assert!(ModelSerializer::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_bad_version() {
        let mut bytes = ModelSerializer::to_bytes(&[vec![1.0]]);
        // Overwrite version byte (index 4)
        bytes[4] = 99;
        assert!(ModelSerializer::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_truncated() {
        let bytes = ModelSerializer::to_bytes(&[vec![1.0, 2.0]]);
        let half = bytes.len() / 2;
        assert!(ModelSerializer::from_bytes(&bytes[..half]).is_err());
    }

    #[test]
    fn test_serialized_len() {
        let weights = vec![vec![1.0_f32; 10], vec![2.0_f32; 5]];
        let bytes = ModelSerializer::to_bytes(&weights);
        assert_eq!(bytes.len(), ModelSerializer::serialized_len(&weights));
    }

    #[test]
    fn test_nan_inf_preserved() {
        let weights = vec![vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY]];
        let bytes = ModelSerializer::to_bytes(&weights);
        let recovered = ModelSerializer::from_bytes(&bytes).unwrap();
        assert!(recovered[0][0].is_nan());
        assert!(recovered[0][1].is_infinite() && recovered[0][1] > 0.0);
        assert!(recovered[0][2].is_infinite() && recovered[0][2] < 0.0);
    }
}
