use oximedia_container::demux::AviMjpegReader;

fn le_u32(value: u32) -> [u8; 4] {
    value.to_le_bytes()
}

fn make_chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(fourcc);
    data.extend_from_slice(&le_u32(payload.len() as u32));
    data.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        data.push(0);
    }
    data
}

fn make_list(list_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"LIST");
    data.extend_from_slice(&le_u32((payload.len() + 4) as u32));
    data.extend_from_slice(list_type);
    data.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        data.push(0);
    }
    data
}

fn build_avi() -> Vec<u8> {
    let strh = make_chunk(
        b"strh",
        &[
            b'v', b'i', b'd', b's', b'M', b'J', b'P', b'G', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ],
    );
    let strf = make_chunk(
        b"strf",
        &[
            40, 0, 0, 0, 64, 1, 0, 0, 240, 0, 0, 0, 1, 0, 24, 0, b'M', b'J', b'P', b'G', 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    );
    let strl = make_list(b"strl", &[strh, strf].concat());
    let hdrl = make_list(b"hdrl", &strl);

    let frame_chunks = vec![
        make_chunk(b"00dc", &[0, 0, 0, 0]),
        make_chunk(b"00dc", &[1, 1, 1, 1]),
        make_chunk(b"00dc", &[2, 2, 2, 2]),
        make_chunk(b"00dc", &[3, 3, 3, 3]),
        make_chunk(b"00dc", &[4, 4, 4, 4]),
    ];
    let movi_payload = frame_chunks.concat();
    let movi = make_list(b"movi", &movi_payload);

    let movi_fourcc_pos = 12 + hdrl.len() + 8;
    let mut idx1_payload = Vec::new();
    let mut chunk_offset = 4u32;
    for frame_index in 0..5u32 {
        idx1_payload.extend_from_slice(b"00dc");
        let flags = if frame_index == 0 || frame_index == 3 {
            0x10u32
        } else {
            0
        };
        idx1_payload.extend_from_slice(&le_u32(flags));
        idx1_payload.extend_from_slice(&le_u32(chunk_offset));
        idx1_payload.extend_from_slice(&le_u32(4));
        chunk_offset += 12;
    }
    let idx1 = make_chunk(b"idx1", &idx1_payload);

    let riff_payload = [
        b"AVI ".as_slice(),
        hdrl.as_slice(),
        movi.as_slice(),
        idx1.as_slice(),
    ]
    .concat();
    let mut riff = Vec::new();
    riff.extend_from_slice(b"RIFF");
    riff.extend_from_slice(&le_u32(riff_payload.len() as u32));
    riff.extend_from_slice(&riff_payload);

    assert!(movi_fourcc_pos > 0);
    riff
}

#[test]
fn avi_sample_accurate_seek() {
    let reader = AviMjpegReader::new(build_avi()).expect("reader");
    let cursor = reader.seek_sample_accurate(4).expect("seek");
    assert_eq!(cursor.sample_index, 3);
    assert_eq!(cursor.skip_samples, 1);
}
