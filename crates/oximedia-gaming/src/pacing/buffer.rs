//! Frame buffer management.
//!
//! Manages frame buffers for smooth playback and low latency.

use crate::{GamingError, GamingResult};
use std::collections::VecDeque;

/// Frame buffer for managing queued frames.
pub struct FrameBuffer<T> {
    config: BufferConfig,
    buffer: VecDeque<T>,
}

/// Buffer configuration.
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Minimum buffer size (frames)
    pub min_size: usize,
    /// Maximum buffer size (frames)
    pub max_size: usize,
    /// Target buffer size (frames)
    pub target_size: usize,
}

impl<T> FrameBuffer<T> {
    /// Create a new frame buffer.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: BufferConfig) -> GamingResult<Self> {
        if config.min_size > config.max_size {
            return Err(GamingError::InvalidConfig(
                "Min size cannot be greater than max size".to_string(),
            ));
        }

        if config.target_size < config.min_size || config.target_size > config.max_size {
            return Err(GamingError::InvalidConfig(
                "Target size must be between min and max".to_string(),
            ));
        }

        let max_size = config.max_size;
        Ok(Self {
            config,
            buffer: VecDeque::with_capacity(max_size),
        })
    }

    /// Push a frame to the buffer.
    ///
    /// # Errors
    ///
    /// Returns error if buffer is full.
    pub fn push(&mut self, frame: T) -> GamingResult<()> {
        if self.buffer.len() >= self.config.max_size {
            return Err(GamingError::InvalidConfig("Buffer is full".to_string()));
        }

        self.buffer.push_back(frame);
        Ok(())
    }

    /// Pop a frame from the buffer.
    pub fn pop(&mut self) -> Option<T> {
        self.buffer.pop_front()
    }

    /// Peek at the next frame without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.buffer.front()
    }

    /// Clear all frames from the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get current buffer size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.config.max_size
    }

    /// Check if buffer is underrunning (below minimum).
    #[must_use]
    pub fn is_underrunning(&self) -> bool {
        self.buffer.len() < self.config.min_size
    }

    /// Get buffer utilization (0.0 to 1.0).
    #[must_use]
    pub fn utilization(&self) -> f64 {
        self.buffer.len() as f64 / self.config.max_size as f64
    }

    /// Get buffer configuration.
    #[must_use]
    pub fn config(&self) -> &BufferConfig {
        &self.config
    }
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            min_size: 2,
            max_size: 10,
            target_size: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let config = BufferConfig::default();
        let buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_invalid_config() {
        let config = BufferConfig {
            min_size: 10,
            max_size: 5,
            target_size: 7,
        };
        let result: Result<FrameBuffer<u32>, _> = FrameBuffer::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_push_pop() {
        let config = BufferConfig::default();
        let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

        buffer.push(1).expect("push should succeed");
        buffer.push(2).expect("push should succeed");
        buffer.push(3).expect("push should succeed");

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_full() {
        let config = BufferConfig {
            min_size: 1,
            max_size: 3,
            target_size: 2,
        };
        let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

        buffer.push(1).expect("push should succeed");
        buffer.push(2).expect("push should succeed");
        buffer.push(3).expect("push should succeed");

        assert!(buffer.is_full());

        let result = buffer.push(4);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_underrun() {
        let config = BufferConfig {
            min_size: 3,
            max_size: 10,
            target_size: 5,
        };
        let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

        assert!(buffer.is_underrunning());

        buffer.push(1).expect("push should succeed");
        buffer.push(2).expect("push should succeed");
        assert!(buffer.is_underrunning());

        buffer.push(3).expect("push should succeed");
        assert!(!buffer.is_underrunning());
    }

    #[test]
    fn test_utilization() {
        let config = BufferConfig {
            min_size: 1,
            max_size: 10,
            target_size: 5,
        };
        let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

        assert_eq!(buffer.utilization(), 0.0);

        for i in 0..5 {
            buffer.push(i).expect("push should succeed");
        }

        assert_eq!(buffer.utilization(), 0.5);
    }

    #[test]
    fn test_peek() {
        let config = BufferConfig::default();
        let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

        buffer.push(42).expect("push should succeed");
        assert_eq!(buffer.peek(), Some(&42));
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_clear() {
        let config = BufferConfig::default();
        let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

        buffer.push(1).expect("push should succeed");
        buffer.push(2).expect("push should succeed");
        buffer.push(3).expect("push should succeed");

        buffer.clear();
        assert!(buffer.is_empty());
    }
}
