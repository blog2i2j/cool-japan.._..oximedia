//! Frame reordering.
//!
//! This module provides functions to fix frame order issues.

use super::detect::Frame;
use crate::Result;

/// Reorder frames to presentation order.
pub fn reorder_to_presentation_order(frames: &mut [Frame]) -> Result<()> {
    frames.sort_by_key(|f| f.pts);
    Ok(())
}

/// Reorder frames to decode order.
pub fn reorder_to_decode_order(frames: &mut [Frame]) -> Result<()> {
    frames.sort_by_key(|f| f.dts);
    Ok(())
}

/// Fix DTS/PTS timestamps.
pub fn fix_dts_pts(frames: &mut [Frame]) -> Result<usize> {
    let mut fixed = 0;

    for frame in frames.iter_mut() {
        if frame.dts > frame.pts {
            // DTS should never be greater than PTS
            frame.dts = frame.pts;
            fixed += 1;
        }
    }

    Ok(fixed)
}

/// Resequence frames.
pub fn resequence_frames(frames: &mut [Frame]) {
    for (i, frame) in frames.iter_mut().enumerate() {
        frame.sequence = i as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reorder_to_presentation_order() {
        let mut frames = vec![
            Frame {
                sequence: 0,
                pts: 200,
                dts: 0,
                data: vec![],
            },
            Frame {
                sequence: 1,
                pts: 100,
                dts: 100,
                data: vec![],
            },
        ];

        reorder_to_presentation_order(&mut frames).expect("reorder should succeed");
        assert_eq!(frames[0].pts, 100);
        assert_eq!(frames[1].pts, 200);
    }

    #[test]
    fn test_fix_dts_pts() {
        let mut frames = vec![Frame {
            sequence: 0,
            pts: 100,
            dts: 200,
            data: vec![],
        }];

        let fixed = fix_dts_pts(&mut frames).expect("DTS/PTS fix should succeed");
        assert_eq!(fixed, 1);
        assert_eq!(frames[0].dts, 100);
    }

    #[test]
    fn test_resequence_frames() {
        let mut frames = vec![
            Frame {
                sequence: 10,
                pts: 0,
                dts: 0,
                data: vec![],
            },
            Frame {
                sequence: 20,
                pts: 100,
                dts: 100,
                data: vec![],
            },
        ];

        resequence_frames(&mut frames);
        assert_eq!(frames[0].sequence, 0);
        assert_eq!(frames[1].sequence, 1);
    }
}
