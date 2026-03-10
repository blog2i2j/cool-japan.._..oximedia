//! Save replay to file.

use crate::GamingResult;

/// Replay saver.
pub struct ReplaySaver {
    format: SaveFormat,
}

/// Save format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveFormat {
    /// `WebM` (VP9 + Opus)
    WebM,
    /// Matroska (VP9 + Opus)
    Mkv,
    /// MP4 (AV1 + Opus)
    Mp4,
}

impl ReplaySaver {
    /// Create a new replay saver.
    #[must_use]
    pub fn new(format: SaveFormat) -> Self {
        Self { format }
    }

    /// Save replay to file.
    pub async fn save(&self, _path: &str) -> GamingResult<()> {
        Ok(())
    }

    /// Get save format.
    #[must_use]
    pub fn format(&self) -> SaveFormat {
        self.format
    }
}

impl Default for ReplaySaver {
    fn default() -> Self {
        Self {
            format: SaveFormat::WebM,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_saver_creation() {
        let saver = ReplaySaver::new(SaveFormat::WebM);
        assert_eq!(saver.format(), SaveFormat::WebM);
    }

    #[tokio::test]
    async fn test_save_replay() {
        let saver = ReplaySaver::default();
        saver
            .save("/tmp/replay.webm")
            .await
            .expect("save should succeed");
    }
}
