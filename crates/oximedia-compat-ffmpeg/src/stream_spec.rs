//! FFmpeg stream specifier parsing.
//!
//! Stream specifiers allow users to target specific streams within a file,
//! for example `"v:0"` (first video stream), `"a:1"` (second audio stream),
//! `"s"` (all subtitle streams), or `"0"` (all streams in file 0).
//!
//! Reference: https://ffmpeg.org/ffmpeg.html#Stream-specifiers-1

/// The media type part of a stream specifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamType {
    /// Video streams (`v`).
    Video,
    /// Audio streams (`a`).
    Audio,
    /// Subtitle streams (`s`).
    Subtitle,
    /// Data streams (`d`).
    Data,
    /// Attachment streams (`t`).
    Attachment,
    /// All stream types (no type discriminator specified).
    Any,
}

impl std::fmt::Display for StreamType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Video => write!(f, "v"),
            Self::Audio => write!(f, "a"),
            Self::Subtitle => write!(f, "s"),
            Self::Data => write!(f, "d"),
            Self::Attachment => write!(f, "t"),
            Self::Any => write!(f, "*"),
        }
    }
}

/// A parsed FFmpeg stream specifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamSpec {
    /// The media type, if specified.
    pub stream_type: StreamType,
    /// Zero-based index within streams of the given type.
    /// `None` means "all streams of this type".
    pub index: Option<usize>,
    /// Program ID (from `p:<pid>:<stream_idx>` form), if specified.
    pub program_id: Option<u32>,
}

impl Default for StreamSpec {
    fn default() -> Self {
        Self {
            stream_type: StreamType::Any,
            index: None,
            program_id: None,
        }
    }
}

impl StreamSpec {
    /// Parse an FFmpeg stream specifier string into a [`StreamSpec`].
    ///
    /// Supported forms:
    /// - `"v"` — all video streams
    /// - `"v:0"` — first video stream
    /// - `"a:1"` — second audio stream
    /// - `"s"` — all subtitle streams
    /// - `"0"` — stream index 0 (type = Any)
    /// - `"p:5:0"` — stream 0 in program 5
    pub fn parse(spec: &str) -> anyhow::Result<Self> {
        let spec = spec.trim();

        if spec.is_empty() {
            return Ok(Self::default());
        }

        // Handle program specifier: `p:<pid>:<stream_idx>`
        if let Some(rest) = spec.strip_prefix("p:") {
            let mut parts = rest.splitn(2, ':');
            let pid_str = parts.next().unwrap_or("0");
            let idx_str = parts.next().unwrap_or("");

            let program_id = pid_str
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("invalid program id in specifier '{}'", spec))?;
            let index =
                if idx_str.is_empty() {
                    None
                } else {
                    Some(idx_str.parse::<usize>().map_err(|_| {
                        anyhow::anyhow!("invalid stream index in specifier '{}'", spec)
                    })?)
                };

            return Ok(Self {
                stream_type: StreamType::Any,
                index,
                program_id: Some(program_id),
            });
        }

        // Split on first `:` to separate type from index.
        let mut parts = spec.splitn(2, ':');
        let type_part = parts.next().unwrap_or("");
        let index_part = parts.next();

        let stream_type = match type_part {
            "v" | "V" => StreamType::Video,
            "a" => StreamType::Audio,
            "s" => StreamType::Subtitle,
            "d" => StreamType::Data,
            "t" => StreamType::Attachment,
            // Pure numeric → stream index with Any type
            n if n.parse::<usize>().is_ok() => {
                let index = n.parse::<usize>().ok();
                return Ok(Self {
                    stream_type: StreamType::Any,
                    index,
                    program_id: None,
                });
            }
            _ => {
                anyhow::bail!(
                    "unknown stream type '{}' in specifier '{}'",
                    type_part,
                    spec
                );
            }
        };

        let index = match index_part {
            None | Some("") => None,
            Some(s) => Some(s.parse::<usize>().map_err(|_| {
                anyhow::anyhow!("invalid stream index '{}' in specifier '{}'", s, spec)
            })?),
        };

        Ok(Self {
            stream_type,
            index,
            program_id: None,
        })
    }

    /// Return `true` if this specifier targets only video streams.
    pub fn is_video(&self) -> bool {
        matches!(self.stream_type, StreamType::Video)
    }

    /// Return `true` if this specifier targets only audio streams.
    pub fn is_audio(&self) -> bool {
        matches!(self.stream_type, StreamType::Audio)
    }

    /// Return `true` if this specifier targets subtitle streams.
    pub fn is_subtitle(&self) -> bool {
        matches!(self.stream_type, StreamType::Subtitle)
    }
}

impl std::fmt::Display for StreamSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(pid) = self.program_id {
            write!(f, "p:{}", pid)?;
            if let Some(idx) = self.index {
                write!(f, ":{}", idx)?;
            }
        } else {
            write!(f, "{}", self.stream_type)?;
            if let Some(idx) = self.index {
                write!(f, ":{}", idx)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_first() {
        let s = StreamSpec::parse("v:0").expect("parse");
        assert_eq!(s.stream_type, StreamType::Video);
        assert_eq!(s.index, Some(0));
    }

    #[test]
    fn test_audio_all() {
        let s = StreamSpec::parse("a").expect("parse");
        assert_eq!(s.stream_type, StreamType::Audio);
        assert_eq!(s.index, None);
    }

    #[test]
    fn test_numeric() {
        let s = StreamSpec::parse("2").expect("parse");
        assert_eq!(s.stream_type, StreamType::Any);
        assert_eq!(s.index, Some(2));
    }

    #[test]
    fn test_program() {
        let s = StreamSpec::parse("p:5:0").expect("parse");
        assert_eq!(s.program_id, Some(5));
        assert_eq!(s.index, Some(0));
    }
}
