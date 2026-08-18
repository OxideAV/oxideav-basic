//! Black-box `WAVE_FORMAT_EXTENSIBLE` interop over tool-generated
//! fixtures.
//!
//! Forward direction: ffmpeg generates EXTENSIBLE WAV files
//! (multi-channel, 24-bit, float) and our demuxer must route the
//! SubFormat GUID / channel mask / valid-bits surfaces correctly.
//! Reverse direction: our muxer's automatically-promoted EXTENSIBLE
//! output must decode byte-exactly through ffmpeg invoked as a
//! black-box validator. Skips silently when ffmpeg isn't installed —
//! same convention as the codec crates' interop tests.

use oxideav_basic::wav::{self, open_wav_demuxer};
use oxideav_core::{
    ChannelLayout, CodecId, CodecParameters, Demuxer, Error, MediaType, Packet, ReadSeek,
    SampleFormat, StreamInfo, TimeBase, WriteSeek,
};
use std::process::Command;

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Generate a short tone WAV through ffmpeg with the requested codec /
/// channel count, returning the file path.
fn ffmpeg_fixture(tag: &str, codec: &str, channels: u32) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("oxideav-basic-wavext-{tag}.wav"));
    let _ = std::fs::remove_file(&tmp);
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.05",
            "-c:a",
            codec,
            "-ac",
            &channels.to_string(),
        ])
        .arg(&tmp)
        .status()
        .expect("ffmpeg invocation failed");
    assert!(status.success(), "ffmpeg failed to produce WAV fixture");
    tmp
}

fn open_fixture(path: &std::path::Path) -> wav::WavDemuxer {
    let f: Box<dyn ReadSeek> = Box::new(std::fs::File::open(path).unwrap());
    open_wav_demuxer(f).expect("tool-generated EXTENSIBLE WAV opens")
}

/// 6-channel 16-bit: the tool writes the EXTENSIBLE escape hatch with
/// the PCM SubFormat and a 5.1 channel mask; our demuxer must route
/// the GUID back to `pcm_s16le` and surface the mask through the core
/// typed layout.
#[test]
fn tool_generated_6ch_s16_routes_and_layouts() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available — skipping wav EXTENSIBLE interop test");
        return;
    }
    let path = ffmpeg_fixture("6ch-s16", "pcm_s16le", 6);
    let dmx = open_fixture(&path);
    assert_eq!(dmx.format_tag(), 0xFFFE, "6ch output must be EXTENSIBLE");
    let s = &dmx.streams()[0];
    assert_eq!(s.params.codec_id, CodecId::new("pcm_s16le"));
    assert_eq!(s.params.channels, Some(6));
    assert_eq!(s.params.sample_rate, Some(48_000));
    // Either 5.1 row of the staged standard-layouts table (back-pair
    // 0x3F or side-pair 0x60F) maps onto the core Surround51.
    let mask = dmx.channel_mask().expect("mask present");
    assert!(
        mask == 0x3F || mask == 0x60F,
        "unexpected 6ch mask 0x{mask:X}"
    );
    assert_eq!(s.params.channel_layout, Some(ChannelLayout::Surround51));
    let _ = std::fs::remove_file(&path);
}

/// 24-bit stereo: whatever `fmt ` flavour the tool picks, the wire
/// codec must resolve by container size to `pcm_s24le`; when the
/// EXTENSIBLE form is used the union must agree (24 valid bits in a
/// 24-bit container).
#[test]
fn tool_generated_s24_stereo_routes_by_container() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available — skipping wav EXTENSIBLE interop test");
        return;
    }
    let path = ffmpeg_fixture("s24-stereo", "pcm_s24le", 2);
    let dmx = open_fixture(&path);
    let s = &dmx.streams()[0];
    assert_eq!(s.params.codec_id, CodecId::new("pcm_s24le"));
    assert_eq!(s.params.channels, Some(2));
    if dmx.format_tag() == 0xFFFE {
        assert_eq!(dmx.valid_bits_per_sample(), Some(24));
    }
    let _ = std::fs::remove_file(&path);
}

/// 8-channel IEEE float: EXTENSIBLE with the IEEE_FLOAT SubFormat and
/// the staged 7.1 mask (`0x63F`) — routes to `pcm_f32le` +
/// `Surround71`.
#[test]
fn tool_generated_8ch_f32_routes_and_layouts() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available — skipping wav EXTENSIBLE interop test");
        return;
    }
    let path = ffmpeg_fixture("8ch-f32", "pcm_f32le", 8);
    let dmx = open_fixture(&path);
    assert_eq!(dmx.format_tag(), 0xFFFE, "8ch output must be EXTENSIBLE");
    let s = &dmx.streams()[0];
    assert_eq!(s.params.codec_id, CodecId::new("pcm_f32le"));
    assert_eq!(s.params.channels, Some(8));
    assert_eq!(dmx.channel_mask(), Some(0x63F));
    assert_eq!(s.params.channel_layout, Some(ChannelLayout::Surround71));
    let _ = std::fs::remove_file(&path);
}

/// Reverse direction: our automatically-promoted EXTENSIBLE output
/// (6-channel 16-bit) must decode byte-exactly through the black-box
/// validator (`-f s16le` raw decode), proving the header the muxer
/// writes describes the interleaved payload correctly.
#[test]
fn our_auto_extensible_output_validates_black_box() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available — skipping wav EXTENSIBLE interop test");
        return;
    }
    // 48 frames of 6 distinct s16 samples.
    let frame: [i16; 6] = [-1000, 2000, -3000, 4000, -5000, 6000];
    let mut payload = Vec::new();
    for _ in 0..48 {
        for s in &frame {
            payload.extend_from_slice(&s.to_le_bytes());
        }
    }

    let mut params = CodecParameters::audio(CodecId::new("pcm_s16le"));
    params.media_type = MediaType::Audio;
    params.channels = Some(6);
    params.sample_rate = Some(48_000);
    params.sample_format = Some(SampleFormat::S16);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 48_000),
        duration: None,
        start_time: Some(0),
        params,
    };

    let tmp = std::env::temp_dir().join("oxideav-basic-wavext-ours-6ch.wav");
    let _ = std::fs::remove_file(&tmp);
    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = wav::open_muxer_with(
            ws,
            std::slice::from_ref(&stream),
            wav::WavMuxOptions::default(),
        )
        .unwrap();
        mux.write_header().unwrap();
        let pkt = Packet::new(0, stream.time_base, payload.clone());
        mux.write_packet(&pkt).unwrap();
        mux.write_trailer().unwrap();
    }

    // The auto-promotion must have produced the EXTENSIBLE form.
    let bytes = std::fs::read(&tmp).unwrap();
    assert_eq!(&bytes[12..16], b"fmt ");
    assert_eq!(
        u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
        40
    );

    // Black-box decode to raw s16le and compare byte-for-byte.
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&tmp)
        .args(["-f", "s16le", "-"])
        .output()
        .expect("ffmpeg invocation failed");
    assert!(
        out.status.success(),
        "validator rejected our EXTENSIBLE output: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        out.stdout, payload,
        "black-box raw decode must match the muxed payload byte-for-byte"
    );

    // And our own demuxer agrees end-to-end.
    let f: Box<dyn ReadSeek> = Box::new(std::fs::File::open(&tmp).unwrap());
    let mut dmx = open_wav_demuxer(f).unwrap();
    assert_eq!(
        dmx.streams()[0].params.channel_layout,
        Some(ChannelLayout::Surround51)
    );
    let mut got = Vec::new();
    loop {
        match dmx.next_packet() {
            Ok(p) => got.extend_from_slice(&p.data),
            Err(Error::Eof) => break,
            Err(e) => panic!("demux error: {e}"),
        }
    }
    assert_eq!(got, payload);
    let _ = std::fs::remove_file(&tmp);
}
