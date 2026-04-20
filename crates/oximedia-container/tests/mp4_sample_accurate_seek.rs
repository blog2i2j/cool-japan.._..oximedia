use bytes::Bytes;
use oximedia_container::demux::Mp4Demuxer;
use oximedia_container::Demuxer;
use oximedia_io::MemorySource;

fn u16be(value: u16) -> [u8; 2] {
    value.to_be_bytes()
}

fn u32be(value: u32) -> [u8; 4] {
    value.to_be_bytes()
}

fn make_box(tag: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let size = 8u32 + content.len() as u32;
    let mut data = Vec::with_capacity(size as usize);
    data.extend_from_slice(&u32be(size));
    data.extend_from_slice(tag);
    data.extend_from_slice(content);
    data
}

fn build_mp4() -> Vec<u8> {
    let sample_count = 5u32;

    let mdhd = {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]);
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(1000));
        content.extend_from_slice(&u32be(5000));
        content.extend_from_slice(&[0x55, 0xC4, 0x00, 0x00]);
        make_box(b"mdhd", &content)
    };

    let hdlr = {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]);
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(b"vide");
        content.extend_from_slice(&[0u8; 12]);
        content.push(0);
        make_box(b"hdlr", &content)
    };

    let stsd = {
        let mut entry = Vec::new();
        entry.extend_from_slice(&[0u8; 6]);
        entry.extend_from_slice(&u16be(1));
        entry.extend_from_slice(&[0u8; 2]);
        entry.extend_from_slice(&[0u8; 2]);
        entry.extend_from_slice(&[0u8; 12]);
        entry.extend_from_slice(&u16be(320));
        entry.extend_from_slice(&u16be(240));
        entry.extend_from_slice(&u32be(0x0048_0000));
        entry.extend_from_slice(&u32be(0x0048_0000));
        entry.extend_from_slice(&u32be(0));
        entry.extend_from_slice(&u16be(1));
        entry.extend_from_slice(&[0u8; 32]);
        entry.extend_from_slice(&u16be(0x0018));
        entry.extend_from_slice(&[0xFF, 0xFF]);

        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]);
        content.extend_from_slice(&u32be(1));
        content.extend_from_slice(&u32be((8 + entry.len()) as u32));
        content.extend_from_slice(b"av01");
        content.extend_from_slice(&entry);
        make_box(b"stsd", &content)
    };

    let stts = {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]);
        content.extend_from_slice(&u32be(1));
        content.extend_from_slice(&u32be(sample_count));
        content.extend_from_slice(&u32be(1000));
        make_box(b"stts", &content)
    };

    let stsc = {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]);
        content.extend_from_slice(&u32be(1));
        content.extend_from_slice(&u32be(1));
        content.extend_from_slice(&u32be(sample_count));
        content.extend_from_slice(&u32be(1));
        make_box(b"stsc", &content)
    };

    let stsz = {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]);
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(sample_count));
        for _ in 0..sample_count {
            content.extend_from_slice(&u32be(4));
        }
        make_box(b"stsz", &content)
    };

    let stco = {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]);
        content.extend_from_slice(&u32be(1));
        content.extend_from_slice(&u32be(0));
        make_box(b"stco", &content)
    };

    let stss = {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]);
        content.extend_from_slice(&u32be(2));
        content.extend_from_slice(&u32be(1));
        content.extend_from_slice(&u32be(4));
        make_box(b"stss", &content)
    };

    let ctts = {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]);
        content.extend_from_slice(&u32be(3));
        content.extend_from_slice(&u32be(1));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(1));
        content.extend_from_slice(&u32be(100));
        content.extend_from_slice(&u32be(3));
        content.extend_from_slice(&u32be(200));
        make_box(b"ctts", &content)
    };

    let stbl = make_box(
        b"stbl",
        &[stsd, stts, stsc, stsz, stco, stss, ctts].concat(),
    );
    let minf = make_box(b"minf", &stbl);
    let mdia = make_box(b"mdia", &[mdhd, hdlr, minf].concat());

    let tkhd = {
        let mut content = Vec::new();
        content.extend_from_slice(&[0x00, 0x00, 0x00, 0x03]);
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(1));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(5000));
        content.extend_from_slice(&[0u8; 8]);
        content.extend_from_slice(&[0u8; 4]);
        content.extend_from_slice(&[0u8; 4]);
        content.extend_from_slice(&u32be(0x0001_0000));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0x0001_0000));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0x4000_0000));
        content.extend_from_slice(&u32be(320u32 << 16));
        content.extend_from_slice(&u32be(240u32 << 16));
        make_box(b"tkhd", &content)
    };

    let trak = make_box(b"trak", &[tkhd, mdia].concat());

    let mvhd = {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]);
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(1000));
        content.extend_from_slice(&u32be(5000));
        content.extend_from_slice(&u32be(0x0001_0000));
        content.extend_from_slice(&u16be(0x0100));
        content.extend_from_slice(&[0u8; 2]);
        content.extend_from_slice(&[0u8; 8]);
        content.extend_from_slice(&u32be(0x0001_0000));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0x0001_0000));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(&u32be(0x4000_0000));
        content.extend_from_slice(&[0u8; 24]);
        content.extend_from_slice(&u32be(2));
        make_box(b"mvhd", &content)
    };

    let ftyp = {
        let mut content = Vec::new();
        content.extend_from_slice(b"isom");
        content.extend_from_slice(&u32be(0));
        content.extend_from_slice(b"isom");
        make_box(b"ftyp", &content)
    };

    let moov = make_box(b"moov", &[mvhd, trak].concat());
    let mut file = Vec::new();
    file.extend_from_slice(&ftyp);
    file.extend_from_slice(&moov);

    let mdat_offset = file.len() as u32 + 8;
    let mut mdat_payload = Vec::new();
    for value in 0..sample_count {
        mdat_payload.extend_from_slice(&[value as u8; 4]);
    }
    file.extend_from_slice(&make_box(b"mdat", &mdat_payload));

    for index in 0..file.len().saturating_sub(16) {
        if &file[index..index + 4] == b"stco" {
            let offset_pos = index + 12;
            file[offset_pos..offset_pos + 4].copy_from_slice(&mdat_offset.to_be_bytes());
            break;
        }
    }

    file
}

#[tokio::test]
async fn mp4_sample_accurate_seek() {
    let source = MemorySource::new(Bytes::from(build_mp4()));
    let mut demuxer = Mp4Demuxer::new(source);
    demuxer.probe().await.expect("probe");

    let cursor = demuxer.seek_sample_accurate(2200).await.expect("seek");
    assert_eq!(cursor.sample_index, 0);
    assert_eq!(cursor.skip_samples, 2);
    assert_eq!(cursor.target_pts, 2200);
}
