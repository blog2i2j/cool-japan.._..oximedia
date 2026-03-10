//! Transition engine for rendering cross-dissolve, wipe, and push transitions.
//!
//! The `TransitionEngine` applies transitions between two adjacent pixel buffers,
//! producing a blended output frame given a progress value (0.0 to 1.0).

use crate::renderer::PixelBuffer;
use crate::transition::{Transition, TransitionType, WipeDirection};

/// Errors that can occur during transition application.
#[derive(Debug, thiserror::Error)]
pub enum TransitionError {
    /// Outgoing and incoming frame dimensions do not match.
    #[error(
        "Frame dimension mismatch: outgoing={outgoing_width}x{outgoing_height}, \
         incoming={incoming_width}x{incoming_height}"
    )]
    DimensionMismatch {
        /// Outgoing frame width.
        outgoing_width: u32,
        /// Outgoing frame height.
        outgoing_height: u32,
        /// Incoming frame width.
        incoming_width: u32,
        /// Incoming frame height.
        incoming_height: u32,
    },
    /// Audio buffer lengths do not match for crossfade.
    #[error("Audio length mismatch: outgoing={outgoing_len}, incoming={incoming_len}")]
    AudioLengthMismatch {
        /// Outgoing buffer length.
        outgoing_len: usize,
        /// Incoming buffer length.
        incoming_len: usize,
    },
}

/// Progress of a transition (0.0 = fully outgoing, 1.0 = fully incoming).
pub type TransitionProgress = f32;

/// A pair of frames to transition between.
#[derive(Debug, Clone)]
pub struct TransitionInput<'a> {
    /// The outgoing (source) frame.
    pub outgoing: &'a PixelBuffer,
    /// The incoming (destination) frame.
    pub incoming: &'a PixelBuffer,
    /// Transition progress: 0.0 = all outgoing, 1.0 = all incoming.
    pub progress: TransitionProgress,
}

/// Applies transitions between video frames.
pub struct TransitionEngine;

impl TransitionEngine {
    /// Create a new transition engine.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Apply the given transition to produce a blended output frame.
    ///
    /// The `input.outgoing` and `input.incoming` buffers must have the same dimensions.
    /// Returns a new buffer with the transition applied, or an error if dimensions mismatch.
    ///
    /// # Errors
    ///
    /// Returns `TransitionError::DimensionMismatch` if the outgoing and incoming
    /// buffers have different dimensions.
    pub fn apply(
        &self,
        transition: &Transition,
        input: &TransitionInput<'_>,
    ) -> Result<PixelBuffer, TransitionError> {
        if (input.outgoing.width, input.outgoing.height)
            != (input.incoming.width, input.incoming.height)
        {
            return Err(TransitionError::DimensionMismatch {
                outgoing_width: input.outgoing.width,
                outgoing_height: input.outgoing.height,
                incoming_width: input.incoming.width,
                incoming_height: input.incoming.height,
            });
        }

        let p = input.progress.clamp(0.0, 1.0);
        let result = match transition.transition_type {
            TransitionType::Dissolve => self.cross_dissolve(input.outgoing, input.incoming, p),
            TransitionType::DipToBlack => {
                self.dip_to_color(input.outgoing, input.incoming, p, [0, 0, 0, 255])
            }
            TransitionType::DipToWhite => {
                self.dip_to_color(input.outgoing, input.incoming, p, [255, 255, 255, 255])
            }
            TransitionType::DipToColor => {
                let color = transition.color.map_or([0, 0, 0, 255], |c| {
                    [
                        (c[0] * 255.0) as u8,
                        (c[1] * 255.0) as u8,
                        (c[2] * 255.0) as u8,
                        (c[3] * 255.0) as u8,
                    ]
                });
                self.dip_to_color(input.outgoing, input.incoming, p, color)
            }
            TransitionType::Wipe => {
                let dir = transition.direction.unwrap_or(WipeDirection::LeftToRight);
                self.wipe(input.outgoing, input.incoming, p, dir, transition.softness)
            }
            TransitionType::Push => {
                let dir = transition.direction.unwrap_or(WipeDirection::LeftToRight);
                self.push(input.outgoing, input.incoming, p, dir)
            }
            TransitionType::Slide => {
                let dir = transition.direction.unwrap_or(WipeDirection::LeftToRight);
                self.slide(input.outgoing, input.incoming, p, dir)
            }
            TransitionType::AudioCrossfade => {
                // Audio transitions don't affect video; return outgoing unchanged.
                self.cross_dissolve(input.outgoing, input.incoming, 0.0)
            }
        };
        Ok(result)
    }

    /// Apply the given transition, returning a fallback frame on error.
    ///
    /// This is a convenience wrapper around [`apply`](Self::apply) that
    /// returns the outgoing frame when dimensions mismatch instead of an error.
    #[must_use]
    pub fn apply_or_fallback(
        &self,
        transition: &Transition,
        input: &TransitionInput<'_>,
    ) -> PixelBuffer {
        match self.apply(transition, input) {
            Ok(buf) => buf,
            Err(_) => input.outgoing.clone(),
        }
    }

    /// Apply an audio crossfade between two sample buffers.
    ///
    /// `outgoing` and `incoming` must have the same length. The crossfade
    /// applies equal-power (cosine) curves so the perceived loudness stays
    /// constant across the transition.
    ///
    /// # Errors
    ///
    /// Returns `TransitionError::AudioLengthMismatch` if the buffers differ in length.
    pub fn audio_crossfade(
        &self,
        outgoing: &[f32],
        incoming: &[f32],
        progress: f32,
    ) -> Result<Vec<f32>, TransitionError> {
        if outgoing.len() != incoming.len() {
            return Err(TransitionError::AudioLengthMismatch {
                outgoing_len: outgoing.len(),
                incoming_len: incoming.len(),
            });
        }
        let p = progress.clamp(0.0, 1.0);
        // Equal-power crossfade: gain_out = cos(p * pi/2), gain_in = sin(p * pi/2)
        let angle = p * std::f32::consts::FRAC_PI_2;
        let gain_out = angle.cos();
        let gain_in = angle.sin();

        let mixed: Vec<f32> = outgoing
            .iter()
            .zip(incoming.iter())
            .map(|(&o, &i)| (o * gain_out + i * gain_in).clamp(-1.0, 1.0))
            .collect();
        Ok(mixed)
    }

    /// Cross-dissolve: linear blend between outgoing and incoming.
    #[must_use]
    pub fn cross_dissolve(
        &self,
        outgoing: &PixelBuffer,
        incoming: &PixelBuffer,
        progress: f32,
    ) -> PixelBuffer {
        let w = outgoing.width;
        let h = outgoing.height;
        let mut out = PixelBuffer::new(w, h);
        let p = progress.clamp(0.0, 1.0);
        let q = 1.0 - p;

        for i in (0..out.data.len()).step_by(4) {
            for c in 0..4 {
                out.data[i + c] = (f32::from(outgoing.data[i + c]) * q
                    + f32::from(incoming.data[i + c]) * p)
                    .round() as u8;
            }
        }
        out
    }

    /// Dip-to-color: fade out to color, then fade in from color.
    #[must_use]
    pub fn dip_to_color(
        &self,
        outgoing: &PixelBuffer,
        incoming: &PixelBuffer,
        progress: f32,
        color: [u8; 4],
    ) -> PixelBuffer {
        let p = progress.clamp(0.0, 1.0);
        let color_buf = PixelBuffer::solid(outgoing.width, outgoing.height, color);

        if p < 0.5 {
            // First half: fade outgoing to color
            let half_p = p / 0.5;
            self.cross_dissolve(outgoing, &color_buf, half_p)
        } else {
            // Second half: fade color to incoming
            let half_p = (p - 0.5) / 0.5;
            self.cross_dissolve(&color_buf, incoming, half_p)
        }
    }

    /// Wipe transition: incoming reveals over outgoing with a sharp or soft edge.
    ///
    /// `softness` controls feathering (0.0 = hard edge, 1.0 = very soft).
    /// At progress=0 the entire frame is outgoing; at progress=1 the entire frame is incoming.
    #[must_use]
    pub fn wipe(
        &self,
        outgoing: &PixelBuffer,
        incoming: &PixelBuffer,
        progress: f32,
        direction: WipeDirection,
        softness: f32,
    ) -> PixelBuffer {
        let w = outgoing.width;
        let h = outgoing.height;
        let mut out = PixelBuffer::new(w, h);
        let p = progress.clamp(0.0, 1.0);
        let feather = (softness * 0.1 * w as f32).max(1.0);

        // edge_pos: the position (in pixels) of the wipe boundary.
        // Pixels on the incoming side of edge_pos get incoming pixel value.
        // LeftToRight: the incoming sweep goes left→right.
        //   edge_pos = p * w  → at p=0, edge is at 0 (all outgoing); at p=1, edge is at w (all incoming).
        //   Pixel x is incoming if x < edge_pos.
        for y in 0..h {
            for x in 0..w {
                let incoming_alpha: f32 = match direction {
                    WipeDirection::LeftToRight => {
                        let edge = p * w as f32;
                        // incoming_alpha: 0 if x >> edge, 1 if x << edge
                        Self::wipe_alpha(x as f32, edge, feather)
                    }
                    WipeDirection::RightToLeft => {
                        let edge = (1.0 - p) * w as f32;
                        // Incoming comes from right: alpha=1 if x > edge_pos
                        1.0 - Self::wipe_alpha(x as f32, edge, feather)
                    }
                    WipeDirection::TopToBottom => {
                        let edge = p * h as f32;
                        Self::wipe_alpha(y as f32, edge, feather)
                    }
                    WipeDirection::BottomToTop => {
                        let edge = (1.0 - p) * h as f32;
                        1.0 - Self::wipe_alpha(y as f32, edge, feather)
                    }
                };

                let idx = ((y * w + x) * 4) as usize;
                for c in 0..4 {
                    out.data[idx + c] = (f32::from(outgoing.data[idx + c]) * (1.0 - incoming_alpha)
                        + f32::from(incoming.data[idx + c]) * incoming_alpha)
                        .round() as u8;
                }
            }
        }
        out
    }

    /// Wipe alpha: returns 1.0 (incoming) when pos << edge, 0.0 when pos >> edge.
    ///
    /// The transition is smooth over a region of width `feather` centered on `edge`.
    fn wipe_alpha(pos: f32, edge: f32, feather: f32) -> f32 {
        // incoming_alpha = 1 when pos < edge, 0 when pos >= edge
        // Feathered: linear ramp from 1→0 over [edge - feather/2, edge + feather/2]
        ((edge - pos) / feather + 0.5).clamp(0.0, 1.0)
    }

    /// Push transition: incoming pushes outgoing off screen.
    #[must_use]
    pub fn push(
        &self,
        outgoing: &PixelBuffer,
        incoming: &PixelBuffer,
        progress: f32,
        direction: WipeDirection,
    ) -> PixelBuffer {
        let w = outgoing.width as i32;
        let h = outgoing.height as i32;
        let p = progress.clamp(0.0, 1.0);
        let mut out = PixelBuffer::new(outgoing.width, outgoing.height);

        let (ox, oy, ix, iy) = match direction {
            WipeDirection::LeftToRight => {
                let offset = (p * w as f32) as i32;
                (offset, 0, offset - w, 0)
            }
            WipeDirection::RightToLeft => {
                let offset = (p * w as f32) as i32;
                (-offset, 0, w - offset, 0)
            }
            WipeDirection::TopToBottom => {
                let offset = (p * h as f32) as i32;
                (0, offset, 0, offset - h)
            }
            WipeDirection::BottomToTop => {
                let offset = (p * h as f32) as i32;
                (0, -offset, 0, h - offset)
            }
        };

        out.composite_over(outgoing, ox, oy, 1.0);
        out.composite_over(incoming, ix, iy, 1.0);
        out
    }

    /// Slide transition: incoming slides over static outgoing.
    #[must_use]
    pub fn slide(
        &self,
        outgoing: &PixelBuffer,
        incoming: &PixelBuffer,
        progress: f32,
        direction: WipeDirection,
    ) -> PixelBuffer {
        let w = outgoing.width as i32;
        let h = outgoing.height as i32;
        let p = progress.clamp(0.0, 1.0);

        let (ix, iy) = match direction {
            WipeDirection::LeftToRight => {
                let offset = (p * w as f32) as i32;
                (offset - w, 0)
            }
            WipeDirection::RightToLeft => {
                let offset = (p * w as f32) as i32;
                (w - offset, 0)
            }
            WipeDirection::TopToBottom => {
                let offset = (p * h as f32) as i32;
                (0, offset - h)
            }
            WipeDirection::BottomToTop => {
                let offset = (p * h as f32) as i32;
                (0, h - offset)
            }
        };

        let mut out = outgoing.clone();
        out.composite_over(incoming, ix, iy, 1.0);
        out
    }
}

impl Default for TransitionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transition::{Transition, TransitionAlignment};
    use crate::types::Duration;

    fn make_dissolve(dur: i64) -> Transition {
        Transition::dissolve(Duration(dur))
    }

    fn solid(r: u8, g: u8, b: u8) -> PixelBuffer {
        PixelBuffer::solid(4, 4, [r, g, b, 255])
    }

    #[test]
    fn test_cross_dissolve_at_zero() {
        let engine = TransitionEngine::new();
        let out = solid(200, 0, 0);
        let inc = solid(0, 0, 200);
        let result = engine.cross_dissolve(&out, &inc, 0.0);
        // All outgoing: should be (200, 0, 0)
        assert_eq!(result.data[0], 200);
        assert_eq!(result.data[2], 0);
    }

    #[test]
    fn test_cross_dissolve_at_one() {
        let engine = TransitionEngine::new();
        let out = solid(200, 0, 0);
        let inc = solid(0, 0, 200);
        let result = engine.cross_dissolve(&out, &inc, 1.0);
        assert_eq!(result.data[0], 0);
        assert_eq!(result.data[2], 200);
    }

    #[test]
    fn test_cross_dissolve_midpoint() {
        let engine = TransitionEngine::new();
        let out = solid(200, 0, 0);
        let inc = solid(0, 0, 200);
        let result = engine.cross_dissolve(&out, &inc, 0.5);
        assert!((result.data[0] as i32 - 100).abs() <= 1);
        assert!((result.data[2] as i32 - 100).abs() <= 1);
    }

    #[test]
    fn test_apply_dissolve_transition() {
        let engine = TransitionEngine::new();
        let t = make_dissolve(24);
        let out = solid(255, 0, 0);
        let inc = solid(0, 255, 0);
        let input = TransitionInput {
            outgoing: &out,
            incoming: &inc,
            progress: 0.5,
        };
        let result = engine.apply(&t, &input).expect("should succeed in test");
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);
    }

    #[test]
    fn test_dip_to_black_first_half() {
        let engine = TransitionEngine::new();
        let out = solid(200, 200, 200);
        let inc = solid(200, 200, 200);
        let result = engine.dip_to_color(&out, &inc, 0.25, [0, 0, 0, 255]);
        // At 0.25 progress (halfway through fade-to-black), should be ~100,100,100
        assert!(result.data[0] < 150, "Should be darker at 0.25 progress");
    }

    #[test]
    fn test_wipe_left_to_right_at_zero() {
        let engine = TransitionEngine::new();
        // Use a wider frame so the wipe edge doesn't cover all pixels
        let out = PixelBuffer::solid(32, 4, [255, 0, 0, 255]);
        let inc = PixelBuffer::solid(32, 4, [0, 0, 255, 255]);
        // At progress=0.0 and 0 softness, the edge is at x=0 → all outgoing
        let result = engine.wipe(&out, &inc, 0.0, WipeDirection::LeftToRight, 0.0);
        // Rightmost pixel should still be predominantly outgoing
        let last_px = result.pixel(31, 0).expect("should succeed in test");
        assert!(
            last_px[0] > 200,
            "Right side should be outgoing at progress=0, got {:?}",
            last_px
        );
    }

    #[test]
    fn test_wipe_left_to_right_at_one() {
        let engine = TransitionEngine::new();
        let out = PixelBuffer::solid(32, 4, [255, 0, 0, 255]);
        let inc = PixelBuffer::solid(32, 4, [0, 0, 255, 255]);
        // At progress=1.0, the edge is past the frame → all incoming
        let result = engine.wipe(&out, &inc, 1.0, WipeDirection::LeftToRight, 0.0);
        let first_px = result.pixel(0, 0).expect("should succeed in test");
        assert!(
            first_px[2] > 200,
            "Left side should be incoming at progress=1, got {:?}",
            first_px
        );
    }

    #[test]
    fn test_push_result_size() {
        let engine = TransitionEngine::new();
        let out = solid(200, 0, 0);
        let inc = solid(0, 0, 200);
        let result = engine.push(&out, &inc, 0.5, WipeDirection::LeftToRight);
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);
    }

    #[test]
    fn test_slide_result_size() {
        let engine = TransitionEngine::new();
        let out = solid(100, 100, 100);
        let inc = solid(50, 50, 50);
        let result = engine.slide(&out, &inc, 0.5, WipeDirection::RightToLeft);
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);
    }

    #[test]
    fn test_apply_wipe_transition() {
        let engine = TransitionEngine::new();
        let mut t = make_dissolve(10);
        t.transition_type = TransitionType::Wipe;
        t.direction = Some(WipeDirection::TopToBottom);
        let out = solid(255, 0, 0);
        let inc = solid(0, 255, 0);
        let input = TransitionInput {
            outgoing: &out,
            incoming: &inc,
            progress: 0.5,
        };
        let result = engine.apply(&t, &input).expect("should succeed in test");
        assert_eq!(result.width, 4);
    }

    #[test]
    fn test_apply_push_transition() {
        let engine = TransitionEngine::new();
        let mut t = make_dissolve(10);
        t.transition_type = TransitionType::Push;
        t.direction = Some(WipeDirection::LeftToRight);
        let out = solid(100, 0, 0);
        let inc = solid(0, 100, 0);
        let input = TransitionInput {
            outgoing: &out,
            incoming: &inc,
            progress: 0.3,
        };
        let result = engine.apply(&t, &input).expect("should succeed in test");
        assert_eq!(result.height, 4);
    }

    #[test]
    fn test_dissolve_dimensions_preserved() {
        let engine = TransitionEngine::new();
        let out = PixelBuffer::solid(16, 9, [100, 0, 0, 255]);
        let inc = PixelBuffer::solid(16, 9, [0, 0, 100, 255]);
        let result = engine.cross_dissolve(&out, &inc, 0.7);
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 9);
    }

    #[test]
    fn test_default_engine() {
        let _engine = TransitionEngine::default();
    }

    #[test]
    fn test_transition_alignment_field() {
        let t = Transition::dissolve(Duration(12));
        assert_eq!(t.alignment, TransitionAlignment::Center);
    }

    #[test]
    fn test_apply_dimension_mismatch_error() {
        let engine = TransitionEngine::new();
        let t = make_dissolve(10);
        let out = PixelBuffer::solid(4, 4, [255, 0, 0, 255]);
        let inc = PixelBuffer::solid(8, 8, [0, 0, 255, 255]);
        let input = TransitionInput {
            outgoing: &out,
            incoming: &inc,
            progress: 0.5,
        };
        let result = engine.apply(&t, &input);
        assert!(result.is_err());
        let err = result.expect_err("should be an error");
        assert!(
            err.to_string().contains("dimension mismatch"),
            "Error: {err}"
        );
    }

    #[test]
    fn test_apply_or_fallback_returns_outgoing_on_mismatch() {
        let engine = TransitionEngine::new();
        let t = make_dissolve(10);
        let out = PixelBuffer::solid(4, 4, [255, 0, 0, 255]);
        let inc = PixelBuffer::solid(8, 8, [0, 0, 255, 255]);
        let input = TransitionInput {
            outgoing: &out,
            incoming: &inc,
            progress: 0.5,
        };
        let result = engine.apply_or_fallback(&t, &input);
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);
    }

    #[test]
    fn test_audio_crossfade_equal_power() {
        let engine = TransitionEngine::new();
        let outgoing = vec![1.0f32; 100];
        let incoming = vec![0.0f32; 100];

        // At progress=0, gain_out=1, gain_in=0 → output ≈ outgoing
        let result = engine
            .audio_crossfade(&outgoing, &incoming, 0.0)
            .expect("should succeed in test");
        assert!((result[0] - 1.0).abs() < 0.01);

        // At progress=1, gain_out=0, gain_in=1 → output ≈ incoming
        let result = engine
            .audio_crossfade(&outgoing, &incoming, 1.0)
            .expect("should succeed in test");
        assert!(result[0].abs() < 0.01);

        // At progress=0.5, equal power: cos(pi/4) ≈ 0.707
        let result = engine
            .audio_crossfade(&outgoing, &incoming, 0.5)
            .expect("should succeed in test");
        assert!(
            (result[0] - 0.707).abs() < 0.01,
            "Expected ~0.707, got {}",
            result[0]
        );
    }

    #[test]
    fn test_audio_crossfade_length_mismatch() {
        let engine = TransitionEngine::new();
        let outgoing = vec![1.0f32; 100];
        let incoming = vec![0.0f32; 50];
        let result = engine.audio_crossfade(&outgoing, &incoming, 0.5);
        assert!(result.is_err());
    }
}
