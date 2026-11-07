//! Every registered PCM wire layout against black-box `ffmpeg -f <fmt>`
//! conversions.
//!
//! A deterministic interleaved stereo signal (extremes, zero, ±1, then
//! a pseudo-random walk) is built in each core sample format. For every
//! codec id, `tests/fixtures/pcm/<id>.raw` holds that signal converted
//! by `ffmpeg -f <core fmt> -i - -f <wire fmt> -` (vendored, so the
//! check runs without ffmpeg). Our decoder must turn the fixture back
//! into the source byte for byte, and our encoder must turn the source
//! into the fixture byte for byte. When ffmpeg is installed the
//! conversion is re-run live and must equal the vendored fixture too.
//!
//! Regenerate the fixtures with
//! `OXIDEAV_PCM_REGEN=1 cargo test --test pcm_layouts_interop`.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, CodecRegistry, Frame, Packet, SampleFormat, TimeBase,
};

const CHANNELS: u16 = 2;
const RATE: u32 = 48_000;
const FRAMES: usize = 40;

fn registry() -> CodecRegistry {
    let mut reg = CodecRegistry::new();
    oxideav_basic::register_codecs(&mut reg);
    reg
}

/// The source signal in `fmt`'s little-endian core bytes.
fn source(fmt: SampleFormat) -> Vec<u8> {
    let n = FRAMES * CHANNELS as usize;
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 32) as u32
    };
    let mut out = Vec::new();
    for i in 0..n {
        let r = next();
        match fmt {
            SampleFormat::U8 => {
                let v = [0u8, 255, 128, 127, 129];
                out.push(*v.get(i).unwrap_or(&(r as u8)));
            }
            SampleFormat::S8 => {
                let v = [i8::MIN, i8::MAX, 0, -1, 1];
                out.push(*v.get(i).unwrap_or(&(r as i8)) as u8);
            }
            SampleFormat::S16 => {
                let v = [i16::MIN, i16::MAX, 0, -1, 1];
                out.extend(v.get(i).copied().unwrap_or(r as i16).to_le_bytes());
            }
            SampleFormat::S24 => {
                let v = [-(1i32 << 23), (1 << 23) - 1, 0, -1, 1];
                // Sign-extended 24-bit value from the top 24 random bits.
                let x = v.get(i).copied().unwrap_or((r as i32) >> 8);
                out.extend(&x.to_le_bytes()[..3]);
            }
            SampleFormat::S32 => {
                let v = [i32::MIN, i32::MAX, 0, -1, 1];
                out.extend(v.get(i).copied().unwrap_or(r as i32).to_le_bytes());
            }
            SampleFormat::F32 => {
                let v = [-1.0f32, 1.0, 0.0, -0.0, f32::MIN_POSITIVE];
                let x = v
                    .get(i)
                    .copied()
                    .unwrap_or((r as i32) as f32 / (1u64 << 31) as f32);
                out.extend(x.to_le_bytes());
            }
            SampleFormat::F64 => {
                let v = [-1.0f64, 1.0, 0.0, -0.0, f64::MIN_POSITIVE];
                let x = v
                    .get(i)
                    .copied()
                    .unwrap_or((r as i32) as f64 / (1u64 << 31) as f64 + 1e-12);
                out.extend(x.to_le_bytes());
            }
            other => panic!("no source for {other:?}"),
        }
    }
    out
}

/// ffmpeg raw-format name of a core interleaved sample format.
fn ffmpeg_core_name(fmt: SampleFormat) -> &'static str {
    match fmt {
        SampleFormat::U8 => "u8",
        SampleFormat::S8 => "s8",
        SampleFormat::S16 => "s16le",
        SampleFormat::S24 => "s24le",
        SampleFormat::S32 => "s32le",
        SampleFormat::F32 => "f32le",
        SampleFormat::F64 => "f64le",
        other => panic!("no ffmpeg name for {other:?}"),
    }
}

fn fixture_path(id: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pcm")
        .join(format!("{id}.raw"))
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Black-box conversion of `src` (core `fmt`) into the wire layout of
/// codec `id`.
fn ffmpeg_convert(fmt: SampleFormat, id: &str, src: &[u8]) -> Vec<u8> {
    let wire = id.strip_prefix("pcm_").unwrap();
    let mut child = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-f"])
        .arg(ffmpeg_core_name(fmt))
        .args(["-ar", &RATE.to_string(), "-ac", &CHANNELS.to_string()])
        .args(["-i", "pipe:0", "-f", wire, "-c:a", id, "pipe:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg");
    child.stdin.take().unwrap().write_all(src).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "ffmpeg failed for {id}");
    out.stdout
}

fn params(id: &str) -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(id));
    p.channels = Some(CHANNELS);
    p.sample_rate = Some(RATE);
    p
}

fn decode(reg: &CodecRegistry, id: &str, wire: &[u8]) -> Vec<u8> {
    let mut d = reg.first_decoder(&params(id)).unwrap();
    assert_eq!(d.output_params_generation(), 0);
    let tb = TimeBase::new(1, RATE as i64);
    d.send_packet(&Packet::new(0, tb, wire.to_vec())).unwrap();
    let Frame::Audio(a) = d.receive_frame().unwrap() else {
        panic!("{id}: audio frame expected")
    };
    assert_eq!(a.samples as usize, FRAMES, "{id}");
    assert_eq!(d.output_params_generation(), 0, "{id}");
    a.data.into_iter().next().unwrap()
}

fn encode(reg: &CodecRegistry, id: &str, core: &[u8]) -> Vec<u8> {
    let mut e = reg.first_encoder(&params(id)).unwrap();
    e.send_frame(&Frame::Audio(AudioFrame {
        samples: FRAMES as u32,
        pts: Some(0),
        data: vec![core.to_vec()],
    }))
    .unwrap();
    e.receive_packet().unwrap().data
}

#[test]
fn every_layout_is_bit_exact_against_the_black_box_conversion() {
    let reg = registry();
    let regen = std::env::var_os("OXIDEAV_PCM_REGEN").is_some();
    let live = ffmpeg_available();
    let mut checked = 0;
    for id in oxideav_basic::pcm::codec_ids() {
        let fmt = oxideav_basic::pcm::sample_format_for(&CodecId::new(id)).unwrap();
        let src = source(fmt);
        let path = fixture_path(id);
        if regen {
            let out = ffmpeg_convert(fmt, id, &src);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, out).unwrap();
        }
        let fixture = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        if live {
            assert_eq!(
                ffmpeg_convert(fmt, id, &src),
                fixture,
                "{id}: live ffmpeg output differs from the vendored fixture"
            );
        }
        assert_eq!(decode(&reg, id, &fixture), src, "{id}: decode");
        assert_eq!(encode(&reg, id, &src), fixture, "{id}: encode");
        checked += 1;
    }
    assert_eq!(checked, 18);
}
