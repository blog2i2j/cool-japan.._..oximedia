//! Timeline reconstruction from matched clips.

use crate::config::ConformConfig;
use crate::error::{ConformError, ConformResult};
use crate::timeline::{Timeline, TimelineClip, Track, TrackKind, Transition};
use crate::types::{ClipMatch, FrameRate, Timecode};
use tracing::{debug, info, warn};

/// Timeline reconstructor.
pub struct TimelineReconstructor {
    /// Configuration.
    config: ConformConfig,
    /// Frame rate.
    fps: FrameRate,
}

impl TimelineReconstructor {
    /// Create a new timeline reconstructor.
    #[must_use]
    pub fn new(fps: FrameRate, config: ConformConfig) -> Self {
        Self { config, fps }
    }

    /// Reconstruct timeline from matched clips.
    ///
    /// # Errors
    ///
    /// Returns an error if timeline reconstruction fails.
    pub fn reconstruct(&self, name: String, matches: &[ClipMatch]) -> ConformResult<Timeline> {
        info!("Reconstructing timeline: {}", name);

        if matches.is_empty() {
            return Err(ConformError::Other("No clips to reconstruct".to_string()));
        }

        let mut timeline = Timeline::new(name, self.fps);

        // Group clips by track
        let (video_clips, audio_clips) = self.group_clips_by_track(matches);

        // Build video tracks
        if !video_clips.is_empty() {
            let video_tracks = self.build_video_tracks(&video_clips)?;
            for track in video_tracks {
                timeline.add_video_track(track);
            }
        }

        // Build audio tracks
        if !audio_clips.is_empty() {
            let audio_tracks = self.build_audio_tracks(&audio_clips)?;
            for track in audio_tracks {
                timeline.add_audio_track(track);
            }
        }

        info!(
            "Timeline reconstructed: {} video tracks, {} audio tracks",
            timeline.video_tracks.len(),
            timeline.audio_tracks.len()
        );

        Ok(timeline)
    }

    /// Group clips by track type.
    fn group_clips_by_track(
        &self,
        matches: &[ClipMatch],
    ) -> (Vec<TimelineClip>, Vec<TimelineClip>) {
        let mut video_clips = Vec::new();
        let mut audio_clips = Vec::new();

        for clip_match in matches {
            let timeline_clip = TimelineClip::from_match(clip_match);

            match clip_match.clip.track {
                crate::types::TrackType::Video => video_clips.push(timeline_clip),
                crate::types::TrackType::Audio => audio_clips.push(timeline_clip),
                crate::types::TrackType::AudioVideo => {
                    video_clips.push(timeline_clip.clone());
                    audio_clips.push(timeline_clip);
                }
            }
        }

        (video_clips, audio_clips)
    }

    /// Build video tracks from clips.
    fn build_video_tracks(&self, clips: &[TimelineClip]) -> ConformResult<Vec<Track>> {
        let mut tracks = Vec::new();
        let mut sorted_clips = clips.to_vec();
        sorted_clips.sort_by_key(|c| c.timeline_in.to_frames(c.fps));

        // For now, create a single video track
        let mut track = Track::new("V1".to_string(), TrackKind::Video);
        track.name = Some("Video 1".to_string());

        for clip in sorted_clips {
            track.add_clip(clip);
        }

        // Add transitions if needed
        self.add_transitions(&mut track)?;

        tracks.push(track);
        Ok(tracks)
    }

    /// Build audio tracks from clips.
    fn build_audio_tracks(&self, clips: &[TimelineClip]) -> ConformResult<Vec<Track>> {
        let mut tracks = Vec::new();
        let mut sorted_clips = clips.to_vec();
        sorted_clips.sort_by_key(|c| c.timeline_in.to_frames(c.fps));

        // For now, create a single audio track
        let mut track = Track::new("A1".to_string(), TrackKind::Audio);
        track.name = Some("Audio 1".to_string());

        for clip in sorted_clips {
            track.add_clip(clip);
        }

        tracks.push(track);
        Ok(tracks)
    }

    /// Add transitions between clips.
    fn add_transitions(&self, track: &mut Track) -> ConformResult<()> {
        if track.clips.len() < 2 {
            return Ok(());
        }

        for i in 0..track.clips.len() - 1 {
            let current = &track.clips[i];
            let next = &track.clips[i + 1];

            // Check for gap or overlap
            let current_out = current.timeline_out.to_frames(current.fps);
            let next_in = next.timeline_in.to_frames(next.fps);

            if current_out == next_in {
                // Perfect cut
                let transition = Transition::cut(current.timeline_out);
                track.add_transition(transition);
            } else if current_out > next_in {
                // Overlap - create dissolve
                let overlap_frames = (current_out - next_in) as u32;
                let transition = Transition::dissolve(next.timeline_in, overlap_frames);
                track.add_transition(transition);
                debug!("Added dissolve transition: {} frames", overlap_frames);
            } else {
                // Gap - warn
                let gap_frames = next_in - current_out;
                warn!("Gap detected: {} frames between clips", gap_frames);
                let transition = Transition::cut(current.timeline_out);
                track.add_transition(transition);
            }
        }

        Ok(())
    }

    /// Validate timeline consistency.
    pub fn validate_timeline(&self, timeline: &Timeline) -> ConformResult<()> {
        for (i, track) in timeline.video_tracks.iter().enumerate() {
            self.validate_track(track, i)?;
        }

        for (i, track) in timeline.audio_tracks.iter().enumerate() {
            self.validate_track(track, i)?;
        }

        Ok(())
    }

    /// Validate a single track.
    fn validate_track(&self, track: &Track, index: usize) -> ConformResult<()> {
        if track.clips.is_empty() {
            warn!("Track {} is empty", index);
            return Ok(());
        }

        // Check for overlapping clips
        for i in 0..track.clips.len() - 1 {
            let current = &track.clips[i];
            let next = &track.clips[i + 1];

            let current_out = current.timeline_out.to_frames(current.fps);
            let next_in = next.timeline_in.to_frames(next.fps);

            if current_out > next_in && self.config.strict_validation {
                return Err(ConformError::Validation(format!(
                    "Overlapping clips in track {}: clip {} ends at {}, clip {} starts at {}",
                    index,
                    i,
                    current.timeline_out,
                    i + 1,
                    next.timeline_in
                )));
            }
        }

        Ok(())
    }

    /// Split timeline into segments.
    pub fn split_into_segments(
        &self,
        timeline: &Timeline,
        segment_duration: f64,
    ) -> ConformResult<Vec<Timeline>> {
        let total_duration = timeline.duration_seconds();
        let num_segments = (total_duration / segment_duration).ceil() as usize;

        let mut segments = Vec::new();

        for i in 0..num_segments {
            let start_time = i as f64 * segment_duration;
            let end_time = ((i + 1) as f64 * segment_duration).min(total_duration);

            let segment = self.extract_segment(
                timeline,
                start_time,
                end_time,
                format!("{}_seg{}", timeline.name, i + 1),
            )?;

            segments.push(segment);
        }

        Ok(segments)
    }

    /// Extract a segment from the timeline.
    fn extract_segment(
        &self,
        timeline: &Timeline,
        start_time: f64,
        end_time: f64,
        name: String,
    ) -> ConformResult<Timeline> {
        let mut segment = Timeline::new(name, timeline.fps);

        let start_tc = crate::utils::seconds_to_timecode(start_time, timeline.fps);
        let end_tc = crate::utils::seconds_to_timecode(end_time, timeline.fps);

        // Extract video tracks
        for track in &timeline.video_tracks {
            let segment_track = self.extract_track_segment(track, start_tc, end_tc)?;
            if !segment_track.clips.is_empty() {
                segment.add_video_track(segment_track);
            }
        }

        // Extract audio tracks
        for track in &timeline.audio_tracks {
            let segment_track = self.extract_track_segment(track, start_tc, end_tc)?;
            if !segment_track.clips.is_empty() {
                segment.add_audio_track(segment_track);
            }
        }

        Ok(segment)
    }

    /// Extract a segment from a track.
    fn extract_track_segment(
        &self,
        track: &Track,
        start_tc: Timecode,
        end_tc: Timecode,
    ) -> ConformResult<Track> {
        let mut segment_track = Track::new(track.id.clone(), track.kind);
        segment_track.name = track.name.clone();

        let start_frames = start_tc.to_frames(self.fps);
        let end_frames = end_tc.to_frames(self.fps);

        for clip in &track.clips {
            let clip_in = clip.timeline_in.to_frames(clip.fps);
            let clip_out = clip.timeline_out.to_frames(clip.fps);

            // Check if clip overlaps with segment
            if clip_out > start_frames && clip_in < end_frames {
                // Clip overlaps - add it (potentially trimmed)
                let mut segment_clip = clip.clone();

                // Trim if needed
                if clip_in < start_frames {
                    let trim_frames = start_frames - clip_in;
                    segment_clip.source_in = Timecode::from_frames(
                        segment_clip.source_in.to_frames(clip.fps) + trim_frames,
                        clip.fps,
                    );
                    segment_clip.timeline_in = start_tc;
                }

                if clip_out > end_frames {
                    let trim_frames = clip_out - end_frames;
                    segment_clip.source_out = Timecode::from_frames(
                        segment_clip.source_out.to_frames(clip.fps) - trim_frames,
                        clip.fps,
                    );
                    segment_clip.timeline_out = end_tc;
                }

                segment_track.add_clip(segment_clip);
            }
        }

        Ok(segment_track)
    }

    /// Merge multiple timelines into one.
    pub fn merge_timelines(&self, timelines: &[Timeline], name: String) -> ConformResult<Timeline> {
        if timelines.is_empty() {
            return Err(ConformError::Other("No timelines to merge".to_string()));
        }

        let fps = timelines[0].fps;
        let mut merged = Timeline::new(name, fps);

        // Merge video tracks
        let max_video_tracks = timelines
            .iter()
            .map(|t| t.video_tracks.len())
            .max()
            .unwrap_or(0);
        for track_idx in 0..max_video_tracks {
            let mut merged_track = Track::new(format!("V{}", track_idx + 1), TrackKind::Video);
            merged_track.name = Some(format!("Video {}", track_idx + 1));

            let mut current_offset = Timecode::new(0, 0, 0, 0);

            for timeline in timelines {
                if let Some(track) = timeline.video_tracks.get(track_idx) {
                    for clip in &track.clips {
                        let mut offset_clip = clip.clone();
                        let offset_frames = current_offset.to_frames(fps);

                        offset_clip.timeline_in = Timecode::from_frames(
                            clip.timeline_in.to_frames(fps) + offset_frames,
                            fps,
                        );
                        offset_clip.timeline_out = Timecode::from_frames(
                            clip.timeline_out.to_frames(fps) + offset_frames,
                            fps,
                        );

                        merged_track.add_clip(offset_clip);
                    }
                }

                // Update offset for next timeline
                let timeline_duration_frames = timeline.duration_frames();
                current_offset = Timecode::from_frames(
                    current_offset.to_frames(fps) + timeline_duration_frames,
                    fps,
                );
            }

            if !merged_track.clips.is_empty() {
                merged.add_video_track(merged_track);
            }
        }

        // Merge audio tracks (similar logic)
        let max_audio_tracks = timelines
            .iter()
            .map(|t| t.audio_tracks.len())
            .max()
            .unwrap_or(0);
        for track_idx in 0..max_audio_tracks {
            let mut merged_track = Track::new(format!("A{}", track_idx + 1), TrackKind::Audio);
            merged_track.name = Some(format!("Audio {}", track_idx + 1));

            let mut current_offset = Timecode::new(0, 0, 0, 0);

            for timeline in timelines {
                if let Some(track) = timeline.audio_tracks.get(track_idx) {
                    for clip in &track.clips {
                        let mut offset_clip = clip.clone();
                        let offset_frames = current_offset.to_frames(fps);

                        offset_clip.timeline_in = Timecode::from_frames(
                            clip.timeline_in.to_frames(fps) + offset_frames,
                            fps,
                        );
                        offset_clip.timeline_out = Timecode::from_frames(
                            clip.timeline_out.to_frames(fps) + offset_frames,
                            fps,
                        );

                        merged_track.add_clip(offset_clip);
                    }
                }

                let timeline_duration_frames = timeline.duration_frames();
                current_offset = Timecode::from_frames(
                    current_offset.to_frames(fps) + timeline_duration_frames,
                    fps,
                );
            }

            if !merged_track.clips.is_empty() {
                merged.add_audio_track(merged_track);
            }
        }

        Ok(merged)
    }

    /// Optimize timeline by removing gaps and overlaps.
    pub fn optimize_timeline(&self, timeline: &mut Timeline) -> ConformResult<()> {
        for track in &mut timeline.video_tracks {
            self.optimize_track(track)?;
        }

        for track in &mut timeline.audio_tracks {
            self.optimize_track(track)?;
        }

        Ok(())
    }

    /// Optimize a single track.
    fn optimize_track(&self, track: &mut Track) -> ConformResult<()> {
        if track.clips.is_empty() {
            return Ok(());
        }

        track.sort_clips();

        // Remove gaps by shifting clips
        let mut current_time = Timecode::new(0, 0, 0, 0);

        for clip in &mut track.clips {
            let clip_duration_frames = clip.duration_frames();

            clip.timeline_in = current_time;
            clip.timeline_out = Timecode::from_frames(
                current_time.to_frames(clip.fps) + clip_duration_frames,
                clip.fps,
            );

            current_time = clip.timeline_out;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ClipReference, MatchMethod, MediaFile, TrackType};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn create_test_match(id: &str, record_in: Timecode, record_out: Timecode) -> ClipMatch {
        let clip = ClipReference {
            id: id.to_string(),
            source_file: Some(format!("{id}.mov")),
            source_in: Timecode::new(1, 0, 0, 0),
            source_out: Timecode::new(1, 0, 10, 0),
            record_in,
            record_out,
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: HashMap::new(),
        };

        ClipMatch {
            clip,
            media: MediaFile::new(PathBuf::from(format!("/test/{id}.mov"))),
            score: 1.0,
            method: MatchMethod::ExactFilename,
            details: String::new(),
        }
    }

    #[test]
    fn test_reconstructor_creation() {
        let reconstructor = TimelineReconstructor::new(FrameRate::Fps25, ConformConfig::default());
        assert_eq!(reconstructor.fps.as_f64(), 25.0);
    }

    #[test]
    fn test_reconstruct_simple_timeline() {
        let reconstructor = TimelineReconstructor::new(FrameRate::Fps25, ConformConfig::default());

        let matches = vec![
            create_test_match(
                "clip1",
                Timecode::new(1, 0, 0, 0),
                Timecode::new(1, 0, 10, 0),
            ),
            create_test_match(
                "clip2",
                Timecode::new(1, 0, 10, 0),
                Timecode::new(1, 0, 20, 0),
            ),
        ];

        let timeline = reconstructor
            .reconstruct("Test Timeline".to_string(), &matches)
            .expect("test expectation failed");

        assert_eq!(timeline.name, "Test Timeline");
        assert_eq!(timeline.video_tracks.len(), 1);
        assert_eq!(timeline.video_tracks[0].clips.len(), 2);
    }

    #[test]
    fn test_validate_timeline() {
        let reconstructor = TimelineReconstructor::new(FrameRate::Fps25, ConformConfig::default());

        let matches = vec![create_test_match(
            "clip1",
            Timecode::new(1, 0, 0, 0),
            Timecode::new(1, 0, 10, 0),
        )];

        let timeline = reconstructor
            .reconstruct("Test".to_string(), &matches)
            .expect("test expectation failed");

        assert!(reconstructor.validate_timeline(&timeline).is_ok());
    }
}
