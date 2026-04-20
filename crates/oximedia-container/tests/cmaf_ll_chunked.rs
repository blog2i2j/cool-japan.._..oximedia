use oximedia_container::mux::cmaf::{CmafConfig, CmafMuxer, CmafSample, CmafTrack, TrackType};

fn top_level_boxes(data: &[u8]) -> Vec<([u8; 4], usize, usize)> {
    let mut boxes = Vec::new();
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let size = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        if size < 8 || offset + size > data.len() {
            break;
        }
        let mut fourcc = [0u8; 4];
        fourcc.copy_from_slice(&data[offset + 4..offset + 8]);
        boxes.push((fourcc, offset, size));
        offset += size;
    }
    boxes
}

#[test]
fn cmaf_ll_chunked() {
    let mut muxer = CmafMuxer::new(CmafConfig {
        low_latency_chunked: true,
        chunk_duration_ms: Some(200),
        ..CmafConfig::default()
    });
    muxer.add_track(CmafTrack {
        track_id: 1,
        track_type: TrackType::Video,
        codec_fourcc: *b"av01",
        timescale: 1000,
        width: Some(1920),
        height: Some(1080),
        sample_rate: None,
        channels: None,
        extradata: Vec::new(),
    });

    let samples: Vec<CmafSample> = (0..5u64)
        .map(|index| CmafSample {
            track_id: 1,
            pts: index * 200,
            dts: index * 200,
            duration: 200,
            data: vec![index as u8; 4],
            keyframe: index == 0,
        })
        .collect();

    let data = muxer.write_media_segment(&samples);
    let boxes = top_level_boxes(&data);
    let styp_count = boxes
        .iter()
        .filter(|(fourcc, _, _)| fourcc == b"styp")
        .count();
    let prft_count = boxes
        .iter()
        .filter(|(fourcc, _, _)| fourcc == b"prft")
        .count();
    let moof_count = boxes
        .iter()
        .filter(|(fourcc, _, _)| fourcc == b"moof")
        .count();
    let mdat_count = boxes
        .iter()
        .filter(|(fourcc, _, _)| fourcc == b"mdat")
        .count();

    assert_eq!(styp_count, 5);
    assert_eq!(prft_count, 5);
    assert_eq!(moof_count, 5);
    assert_eq!(mdat_count, 5);

    for (_, offset, size) in boxes.iter().filter(|(fourcc, _, _)| fourcc == b"styp") {
        let payload_start = offset + 8;
        let payload_end = offset + size;
        assert!(payload_end >= payload_start + 4);
        assert_eq!(&data[payload_start..payload_start + 4], b"cmfl");
    }
}
