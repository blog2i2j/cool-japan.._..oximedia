//! Missing packet interpolation.
//!
//! This module provides functions to interpolate data for missing packets.

use super::recover::{Packet, PacketStatus};
use crate::Result;

/// Interpolate missing packet data.
pub fn interpolate_packet_data(packets: &mut [Packet]) -> Result<usize> {
    let mut interpolated = 0;

    for i in 1..packets.len() - 1 {
        if packets[i].status == PacketStatus::Missing {
            let prev_data = &packets[i - 1].data;
            let next_data = &packets[i + 1].data;

            if !prev_data.is_empty() && !next_data.is_empty() {
                packets[i].data = interpolate_data(prev_data, next_data);
                interpolated += 1;
            }
        }
    }

    Ok(interpolated)
}

/// Interpolate between two data buffers.
fn interpolate_data(prev: &[u8], next: &[u8]) -> Vec<u8> {
    let len = prev.len().min(next.len());
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        let avg = ((prev[i] as u16 + next[i] as u16) / 2) as u8;
        result.push(avg);
    }

    result
}

/// Interpolate using linear prediction.
pub fn interpolate_linear(packets: &mut [Packet], window_size: usize) -> Result<usize> {
    let mut interpolated = 0;

    for i in window_size..packets.len() {
        if packets[i].status == PacketStatus::Missing {
            // Use previous packets to predict
            if let Some(data) = predict_packet_data(packets, i, window_size) {
                packets[i].data = data;
                interpolated += 1;
            }
        }
    }

    Ok(interpolated)
}

/// Predict packet data based on previous packets.
fn predict_packet_data(packets: &[Packet], index: usize, window_size: usize) -> Option<Vec<u8>> {
    let start = index.saturating_sub(window_size);
    let prev_packets: Vec<&Packet> = packets[start..index]
        .iter()
        .filter(|p| p.status == PacketStatus::Valid)
        .collect();

    if prev_packets.is_empty() {
        return None;
    }

    // Simple prediction: use last valid packet
    prev_packets.last().map(|p| p.data.clone())
}

/// Interpolate audio data with fading.
pub fn interpolate_audio_fade(prev: &[u8], next: &[u8]) -> Vec<u8> {
    let len = prev.len().min(next.len());
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        let t = i as f32 / len as f32;
        let value = (prev[i] as f32 * (1.0 - t) + next[i] as f32 * t) as u8;
        result.push(value);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_data() {
        let prev = vec![0, 0, 0];
        let next = vec![100, 100, 100];
        let result = interpolate_data(&prev, &next);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 50);
        assert_eq!(result[1], 50);
        assert_eq!(result[2], 50);
    }

    #[test]
    fn test_interpolate_audio_fade() {
        let prev = vec![0, 0, 0, 0];
        let next = vec![100, 100, 100, 100];
        let result = interpolate_audio_fade(&prev, &next);

        assert_eq!(result.len(), 4);
        assert!(result[0] < result[3]); // Should fade from prev to next
    }

    #[test]
    fn test_predict_packet_data() {
        let packets = vec![
            Packet {
                sequence: 0,
                data: vec![1, 2, 3],
                timestamp: 0,
                status: PacketStatus::Valid,
            },
            Packet {
                sequence: 1,
                data: vec![4, 5, 6],
                timestamp: 100,
                status: PacketStatus::Valid,
            },
        ];

        let predicted = predict_packet_data(&packets, 2, 2);
        assert!(predicted.is_some());
        assert_eq!(
            predicted.expect("expected predicted to be Some/Ok"),
            vec![4, 5, 6]
        );
    }
}
