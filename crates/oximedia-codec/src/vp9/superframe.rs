//! VP9 Superframe handling.
//!
//! A VP9 superframe contains multiple frames packed together with an index.

use crate::error::{CodecError, CodecResult};

/// VP9 Superframe container.
#[derive(Clone, Debug)]
pub struct Superframe {
    /// Individual frames within the superframe.
    pub frames: Vec<Vec<u8>>,
}

/// Superframe index information.
#[derive(Clone, Debug)]
pub struct SuperframeIndex {
    /// Number of frames in the superframe.
    pub frame_count: usize,
    /// Size of each frame in bytes.
    pub frame_sizes: Vec<usize>,
    /// Number of bytes used to encode each frame size.
    pub bytes_per_size: usize,
    /// Total size of the index in bytes.
    pub index_size: usize,
}

impl Superframe {
    /// Parses a superframe from data.
    ///
    /// # Errors
    ///
    /// Returns error if the superframe index is invalid.
    pub fn parse(data: &[u8]) -> CodecResult<Self> {
        if data.is_empty() {
            return Err(CodecError::InvalidBitstream("Empty data".into()));
        }

        if let Some(index) = SuperframeIndex::parse(data)? {
            let mut frames = Vec::with_capacity(index.frame_count);
            let mut offset = 0;

            for &size in &index.frame_sizes {
                if offset + size > data.len() - index.index_size {
                    return Err(CodecError::InvalidBitstream(
                        "Frame size exceeds data".into(),
                    ));
                }
                frames.push(data[offset..offset + size].to_vec());
                offset += size;
            }

            Ok(Self { frames })
        } else {
            Ok(Self {
                frames: vec![data.to_vec()],
            })
        }
    }

    /// Returns true if this contains multiple frames.
    #[must_use]
    pub fn is_superframe(&self) -> bool {
        self.frames.len() > 1
    }

    /// Returns the number of frames in this superframe.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns an iterator over the frames.
    pub fn iter(&self) -> impl Iterator<Item = &[u8]> {
        self.frames.iter().map(Vec::as_slice)
    }
}

impl SuperframeIndex {
    const MARKER: u8 = 0b110;
    const MAX_FRAMES: usize = 8;

    /// Parses a superframe index from data.
    ///
    /// # Errors
    ///
    /// Returns error if the index is malformed.
    pub fn parse(data: &[u8]) -> CodecResult<Option<Self>> {
        if data.is_empty() {
            return Ok(None);
        }

        let marker = data[data.len() - 1];
        if (marker >> 5) != Self::MARKER {
            return Ok(None);
        }

        let bytes_per_size = ((marker >> 3) & 0x3) as usize + 1;
        let frame_count = (marker & 0x7) as usize + 1;

        if frame_count > Self::MAX_FRAMES {
            return Err(CodecError::InvalidBitstream(
                "Too many frames in superframe".into(),
            ));
        }

        let index_size = 1 + frame_count * bytes_per_size + 1;

        if data.len() < index_size {
            return Err(CodecError::InvalidBitstream(
                "Data too short for superframe index".into(),
            ));
        }

        let index_start = data.len() - index_size;
        if data[index_start] != marker {
            return Err(CodecError::InvalidBitstream(
                "Superframe index marker mismatch".into(),
            ));
        }

        let mut frame_sizes = Vec::with_capacity(frame_count);
        let mut offset = index_start + 1;

        for _ in 0..frame_count {
            let mut size: usize = 0;
            for i in 0..bytes_per_size {
                size |= (data[offset + i] as usize) << (i * 8);
            }
            frame_sizes.push(size);
            offset += bytes_per_size;
        }

        Ok(Some(Self {
            frame_count,
            frame_sizes,
            bytes_per_size,
            index_size,
        }))
    }

    /// Returns the total size of all frames.
    #[must_use]
    pub fn total_frame_size(&self) -> usize {
        self.frame_sizes.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_superframe() {
        let data = [0x00, 0x01, 0x02, 0x03];
        let sf = Superframe::parse(&data).expect("should succeed");
        assert!(!sf.is_superframe());
        assert_eq!(sf.frame_count(), 1);
    }

    #[test]
    fn test_empty_data() {
        let data: [u8; 0] = [];
        assert!(Superframe::parse(&data).is_err());
    }

    #[test]
    fn test_superframe_marker() {
        let data = [0x00, 0x01, 0x00, 0x01, 0x02, 0xC1, 0x02, 0x03, 0xC1];
        let index = SuperframeIndex::parse(&data)
            .expect("should succeed")
            .expect("should succeed");
        assert_eq!(index.frame_count, 2);
        assert_eq!(index.frame_sizes, vec![2, 3]);
    }

    #[test]
    fn test_superframe_parse() {
        let data = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xC1, 0x02, 0x03, 0xC1];
        let sf = Superframe::parse(&data).expect("should succeed");
        assert!(sf.is_superframe());
        assert_eq!(sf.frame_count(), 2);
    }
}
