//! Hostile-input sweep: mutated / truncated WAV, `.sln` and Y4M bytes
//! through the demuxers and on into this crate's decoders must fail
//! with an error, never panic, and never hand a decoder a frame whose
//! size disagrees with its stated `output_params()`.

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};

use oxideav_core::{
    CodecRegistry, ContainerRegistry, Error, Frame, NullCodecResolver, ReadSeek, SampleFormat,
};

fn wav_bytes(fmt_tag: u16, channels: u16, bits: u16, extensible: bool) -> Vec<u8> {
    let rate = 8_000u32;
    let block = channels * bits / 8;
    let mut fmt = Vec::new();
    fmt.extend((if extensible { 0xFFFE } else { fmt_tag }).to_le_bytes());
    fmt.extend(channels.to_le_bytes());
    fmt.extend(rate.to_le_bytes());
    fmt.extend((rate * block as u32).to_le_bytes());
    fmt.extend(block.to_le_bytes());
    fmt.extend(bits.to_le_bytes());
    if extensible {
        fmt.extend(22u16.to_le_bytes());
        fmt.extend(bits.to_le_bytes());
        fmt.extend(3u32.to_le_bytes());
        // KSDATAFORMAT_SUBTYPE: tag in the first two bytes, then the
        // fixed base GUID tail.
        fmt.extend(fmt_tag.to_le_bytes());
        fmt.extend([
            0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
        ]);
    }
    let data: Vec<u8> = (0..block as usize * 64).map(|i| (i * 37) as u8).collect();
    let mut info = Vec::new();
    info.extend(b"INFO");
    info.extend(b"INAM");
    info.extend(6u32.to_le_bytes());
    info.extend(b"title\0");
    let mut body = Vec::new();
    body.extend(b"WAVE");
    body.extend(b"fmt ");
    body.extend((fmt.len() as u32).to_le_bytes());
    body.extend(&fmt);
    body.extend(b"LIST");
    body.extend((info.len() as u32).to_le_bytes());
    body.extend(&info);
    body.extend(b"data");
    body.extend((data.len() as u32).to_le_bytes());
    body.extend(&data);
    let mut out = Vec::new();
    out.extend(b"RIFF");
    out.extend((body.len() as u32).to_le_bytes());
    out.extend(body);
    out
}

/// A PCM WAV carrying every metadata chunk the demuxer parses, with
/// small-valued pseudo-random bodies (small so counts and offsets land
/// in range often enough for the parsers to go deep).
fn wav_with_chunks(rng: &mut Rng, rf64: bool) -> Vec<u8> {
    const IDS: &[&[u8; 4]] = &[
        b"bext", b"cue ", b"smpl", b"inst", b"acid", b"axml", b"bxml", b"sxml", b"chna", b"fact",
        b"iXML", b"id3 ", b"plst", b"slnt", b"wavl", b"DISP", b"PAD ", b"JUNK",
    ];
    let mut chunks: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for id in IDS {
        let len = 8 + rng.below(96);
        let body: Vec<u8> = (0..len)
            .map(|_| {
                let r = rng.next() as u8;
                if r & 0x80 != 0 { r & 0x03 } else { r }
            })
            .collect();
        chunks.push((id.to_vec(), body));
    }
    let mut adtl = b"adtl".to_vec();
    for sub in [b"labl", b"note", b"ltxt", b"file"] {
        adtl.extend(sub);
        adtl.extend(12u32.to_le_bytes());
        adtl.extend(1u32.to_le_bytes());
        adtl.extend(b"label\0\0\0");
    }
    chunks.push((b"LIST".to_vec(), adtl));
    let plain = wav_bytes(1, 2, 16, false);
    // Splice the chunks in after `fmt ` (12 + 8 + 16 bytes in).
    let mut body = plain[8..36].to_vec();
    for (id, b) in &chunks {
        body.extend(id);
        body.extend((b.len() as u32).to_le_bytes());
        body.extend(b);
        if b.len() % 2 == 1 {
            body.push(0);
        }
    }
    body.extend(&plain[36..]);
    let mut out = Vec::new();
    if rf64 {
        out.extend(b"RF64");
        out.extend(0xFFFF_FFFFu32.to_le_bytes());
        out.extend(&body[..4]);
        out.extend(b"ds64");
        out.extend(28u32.to_le_bytes());
        out.extend(((body.len() + 36) as u64).to_le_bytes());
        out.extend(256u64.to_le_bytes());
        out.extend(64u64.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(&body[4..]);
    } else {
        out.extend(b"RIFF");
        out.extend((body.len() as u32).to_le_bytes());
        out.extend(body);
    }
    out
}

fn y4m_bytes() -> Vec<u8> {
    let mut out = b"YUV4MPEG2 W6 H4 F30:1 Ip A1:1 C420jpeg XYSCSS=420JPEG\n".to_vec();
    for f in 0..3u8 {
        out.extend(b"FRAME\n");
        out.extend((0..(24 + 6 + 6)).map(|i| i as u8 ^ f));
    }
    out
}

/// Mutations per seed (1.3 M ran clean locally in release mode).
const ITERATIONS: usize = if cfg!(miri) { 20 } else { 3_000 };

/// Deterministic xorshift.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn mutate(rng: &mut Rng, src: &[u8]) -> Vec<u8> {
    let mut v = src.to_vec();
    match rng.below(4) {
        0 => {
            for _ in 0..1 + rng.below(4) {
                let i = rng.below(v.len());
                v[i] = rng.next() as u8;
            }
        }
        1 => v.truncate(rng.below(v.len())),
        2 => {
            // Overwrite four bytes (size fields) with an extreme value.
            let i = rng.below(v.len().saturating_sub(4));
            let x = [0u32, 1, 0x7FFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFE][rng.below(5)];
            v[i..i + 4].copy_from_slice(&x.to_le_bytes());
        }
        _ => {
            let i = rng.below(v.len().saturating_sub(2));
            let x = [0u16, 1, 0xFFFF, 0x8000][rng.below(4)];
            v[i..i + 2].copy_from_slice(&x.to_le_bytes());
        }
    }
    v
}

/// Demux to the end and decode every packet; errors are fine.
/// Returns the number of decoded frames.
fn run(containers: &ContainerRegistry, codecs: &CodecRegistry, fmt: &str, bytes: Vec<u8>) -> usize {
    let mut frames = 0;
    let input: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let Ok(mut dmx) = containers.open_demuxer(fmt, input, &NullCodecResolver) else {
        return 0;
    };
    let streams = dmx.streams().to_vec();
    let mut decoders: Vec<_> = streams
        .iter()
        .map(|s| codecs.first_decoder(&s.params).ok())
        .collect();
    for _ in 0..10_000 {
        let pkt = match dmx.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        let idx = pkt.stream_index as usize;
        let Some(Some(dec)) = decoders.get_mut(idx) else {
            continue;
        };
        if dec.send_packet(&pkt).is_err() {
            continue;
        }
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    frames += 1;
                    let p = dec.output_params().expect("PCM decoder states its output");
                    let fmt: SampleFormat = p.sample_format.unwrap();
                    let want =
                        a.samples as usize * fmt.bytes_per_sample() * p.channels.unwrap() as usize;
                    assert_eq!(a.data[0].len(), want, "audio frame vs output_params");
                }
                Ok(Frame::Video(v)) => {
                    frames += 1;
                    let p = dec.output_params().expect("rawvideo states its output");
                    assert!(v.planes[0].stride as u32 >= p.width.unwrap());
                }
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(_) => break,
            }
        }
    }
    frames
}

#[test]
fn mutated_containers_never_panic() {
    let mut containers = ContainerRegistry::new();
    oxideav_basic::register_containers(&mut containers);
    let mut codecs = CodecRegistry::new();
    oxideav_basic::register_codecs(&mut codecs);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut seeds: Vec<(&str, Vec<u8>)> = vec![
        ("wav", wav_bytes(1, 2, 16, false)),
        ("wav", wav_bytes(1, 1, 24, true)),
        ("wav", wav_bytes(3, 2, 32, false)),
        ("wav", wav_bytes(3, 1, 64, true)),
        ("wav", wav_bytes(1, 1, 8, false)),
        ("slin", (0..200u8).collect()),
        ("y4m", y4m_bytes()),
    ];
    for i in 0..6 {
        seeds.push(("wav", wav_with_chunks(&mut rng, i % 2 == 1)));
    }
    for (fmt, seed) in &seeds {
        // The seeds themselves must go through cleanly.
        assert!(
            run(&containers, &codecs, fmt, seed.clone()) > 0,
            "{fmt} seed"
        );
        for i in 0..ITERATIONS {
            let bytes = mutate(&mut rng, seed);
            let copy = bytes.clone();
            let r = catch_unwind(AssertUnwindSafe(|| {
                run(&containers, &codecs, fmt, bytes);
            }));
            if r.is_err() {
                panic!("{fmt} mutation #{i} panicked; input = {copy:02x?}");
            }
        }
    }
}
