use bytes::Bytes;
use oximedia_container::demux::matroska::matroska_v4::BlockAdditionMapping;
use oximedia_container::demux::MatroskaDemuxer;
use oximedia_container::mux::{MatroskaMuxer, Muxer, MuxerConfig};
use oximedia_container::{CodecParams, Demuxer, StreamInfo};
use oximedia_core::{CodecId, Rational};
use oximedia_io::MemorySource;

#[tokio::test]
async fn mkv_blockaddmapping_roundtrip() {
    let sink = MemorySource::new_writable(4096);
    let mut muxer = MatroskaMuxer::new(sink, MuxerConfig::default());

    let mut stream = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 1000));
    stream.codec_params = CodecParams::video(640, 360);
    stream.codec_params.block_addition_mappings = vec![
        BlockAdditionMapping {
            id_name: Some("type-4".to_string()),
            id_type: Some(4),
            id_extra_data: vec![0x10, 0x20],
        },
        BlockAdditionMapping {
            id_name: Some("type-5".to_string()),
            id_type: Some(5),
            id_extra_data: vec![0x30, 0x40, 0x50],
        },
    ];

    muxer.add_stream(stream).expect("add stream");
    muxer.write_header().await.expect("write header");
    muxer.write_trailer().await.expect("write trailer");

    let sink = muxer.into_sink();
    let source = MemorySource::new(Bytes::copy_from_slice(sink.written_data()));
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe");

    let tracks = demuxer.tracks();
    assert_eq!(tracks.len(), 1);
    assert_eq!(tracks[0].block_addition_mappings.len(), 2);
    assert_eq!(tracks[0].block_addition_mappings[0].id_type, Some(4));
    assert_eq!(
        tracks[0].block_addition_mappings[0].id_name.as_deref(),
        Some("type-4")
    );
    assert_eq!(tracks[0].block_addition_mappings[1].id_type, Some(5));
    assert_eq!(
        tracks[0].block_addition_mappings[1].id_name.as_deref(),
        Some("type-5")
    );
    assert_eq!(
        tracks[0].block_addition_mappings[1].id_extra_data,
        vec![0x30, 0x40, 0x50]
    );
}
