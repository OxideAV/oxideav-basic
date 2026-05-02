//! End-to-end round-trip through the public `Y4mMuxer` →
//! `Y4mDemuxer` pair via the [`ContainerRegistry`].
//!
//! Builds a deterministic 16×16 yuv420p clip with three frames,
//! writes it to a temp file, reads it back through the registry, and
//! asserts byte-exact equality of each frame.

use oxideav_basic::register_containers;
use oxideav_core::{
    CodecId, CodecParameters, ContainerRegistry, Error, MediaType, NullCodecResolver, Packet,
    PixelFormat, Rational, ReadSeek, StreamInfo, TimeBase, WriteSeek,
};

fn synth_yuv420p(w: usize, h: usize, frame: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h + (w / 2) * (h / 2) * 2);
    for y in 0..h {
        for x in 0..w {
            v.push((((x * 5) ^ (y * 7) ^ (frame * 11)) & 0xff) as u8);
        }
    }
    for cy in 0..(h / 2) {
        for cx in 0..(w / 2) {
            v.push((((cx * 13) ^ (cy * 17) ^ (frame * 19)) & 0xff) as u8);
        }
    }
    for cy in 0..(h / 2) {
        for cx in 0..(w / 2) {
            v.push((((cx * 23) ^ (cy * 29) ^ (frame * 31)) & 0xff) as u8);
        }
    }
    v
}

fn make_stream(w: u32, h: u32) -> StreamInfo {
    let mut params = CodecParameters::video(CodecId::new("rawvideo"));
    params.media_type = MediaType::Video;
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));
    StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 30),
        duration: None,
        start_time: Some(0),
        params,
    }
}

#[test]
fn registry_round_trip_16x16_three_frames() {
    let mut reg = ContainerRegistry::new();
    register_containers(&mut reg);

    let stream = make_stream(16, 16);
    let frames: Vec<Vec<u8>> = (0..3).map(|i| synth_yuv420p(16, 16, i)).collect();

    let tmp = std::env::temp_dir().join("oxideav-basic-y4m-roundtrip.y4m");
    let _ = std::fs::remove_file(&tmp);

    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = reg
            .open_muxer("y4m", ws, std::slice::from_ref(&stream))
            .unwrap();
        mux.write_header().unwrap();
        for frame in &frames {
            let pkt = Packet::new(0, stream.time_base, frame.clone());
            mux.write_packet(&pkt).unwrap();
        }
        mux.write_trailer().unwrap();
    }

    let f: Box<dyn ReadSeek> = Box::new(std::fs::File::open(&tmp).unwrap());
    let mut dmx = reg.open_demuxer("y4m", f, &NullCodecResolver).unwrap();
    let s = &dmx.streams()[0];
    assert_eq!(s.params.width, Some(16));
    assert_eq!(s.params.height, Some(16));
    assert_eq!(s.params.codec_id, CodecId::new("rawvideo"));

    for (i, want) in frames.iter().enumerate() {
        let p = dmx.next_packet().unwrap();
        assert_eq!(p.pts, Some(i as i64));
        assert_eq!(&p.data, want, "frame {i} byte mismatch");
    }
    assert!(matches!(dmx.next_packet(), Err(Error::Eof)));

    let _ = std::fs::remove_file(&tmp);
}
