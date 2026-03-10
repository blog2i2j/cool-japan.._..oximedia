//! `SubStation` Alpha (SSA) format parser and writer

use crate::error::Result;
use crate::formats::{FormatParser, FormatWriter};
use crate::types::CaptionTrack;

/// SSA format parser (similar to ASS but v4.00)
pub struct SsaParser;

impl FormatParser for SsaParser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        // SSA is very similar to ASS, reuse ASS parser with minor adjustments
        let ass_parser = super::ass::AssParser;
        ass_parser.parse(data)
    }
}

/// SSA format writer
pub struct SsaWriter;

impl FormatWriter for SsaWriter {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        // Write as SSA v4.00 format (similar to ASS but simpler)
        let mut output = String::new();

        output.push_str("[Script Info]\n");
        output.push_str("ScriptType: v4.00\n");
        if let Some(title) = &track.metadata.title {
            output.push_str(&format!("Title: {title}\n"));
        }
        output.push('\n');

        output.push_str("[V4 Styles]\n");
        output.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding\n");
        output
            .push_str("Style: Default,Arial,48,16777215,16777215,0,0,0,0,1,2,2,2,10,10,10,0,1\n\n");

        output.push_str("[Events]\n");
        output.push_str(
            "Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
        );

        for caption in &track.captions {
            let start = format_timestamp(caption.start);
            let end = format_timestamp(caption.end);
            let speaker = caption.speaker.as_deref().unwrap_or("");
            let text = caption.text.replace('\n', "\\N");

            output.push_str(&format!(
                "Dialogue: Marked=0,{start},{end},Default,{speaker},0,0,0,,{text}\n"
            ));
        }

        Ok(output.into_bytes())
    }
}

fn format_timestamp(ts: crate::types::Timestamp) -> String {
    let (h, m, s, ms) = ts.as_hmsm();
    let centis = ms / 10;
    format!("{h}:{m:02}:{s:02}.{centis:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Caption, Language, Timestamp};

    #[test]
    fn test_write_ssa() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let writer = SsaWriter;
        let output = writer.write(&track).expect("writing should succeed");
        let text = String::from_utf8(output).expect("output should be valid UTF-8");

        assert!(text.contains("[Script Info]"));
        assert!(text.contains("v4.00"));
    }
}
