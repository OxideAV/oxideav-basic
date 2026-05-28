//! Integration test for the WAV `cue ` + `LIST adtl` chunk parser.
//!
//! Layout per `docs/container/riff/metadata/microsoft-riffmci.pdf`
//! chapter 3 §§"Cue-Points Chunk", "Associated Data Chunk".
//!
//! We synthesise a minimal RIFF/WAVE file in memory carrying:
//!
//! - One `fmt ` chunk (mono S16LE @ 8 kHz, the smallest valid WAVE form).
//! - Two `cue ` records (`dwName` = 1 and 7 — non-contiguous to verify
//!   we key on the spec's "unique cue-point ID", not the array index).
//! - One `LIST adtl` containing a `labl` for cue 1, a `note` for cue 7,
//!   and an `ltxt` for cue 1 (length = 480 samples, purpose = `scrp`,
//!   ZSTR = "caption").
//! - One `data` chunk with 4 samples (the minimum so the demuxer doesn't
//!   reject the form).
//!
//! Then we open it through the public `register_containers` API + assert
//! that `Demuxer::metadata` carries the expected `wav:cue.*` keys.

use oxideav_basic::{register_codecs, register_containers};
use oxideav_core::{CodecRegistry, ContainerRegistry, ReadSeek};
use std::io::Cursor;

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_fourcc(out: &mut Vec<u8>, id: &[u8; 4]) {
    out.extend_from_slice(id);
}

/// Push a single 24-byte `<cue-point>` record per the Microsoft RIFF
/// MCI §"Cue-Points Chunk" struct: `dwName + dwPosition + fccChunk +
/// dwChunkStart + dwBlockStart + dwSampleOffset`.
fn push_cue_record(
    out: &mut Vec<u8>,
    name: u32,
    position: u32,
    fcc: &[u8; 4],
    chunk_start: u32,
    block_start: u32,
    sample_offset: u32,
) {
    put_u32(out, name);
    put_u32(out, position);
    put_fourcc(out, fcc);
    put_u32(out, chunk_start);
    put_u32(out, block_start);
    put_u32(out, sample_offset);
}

/// Build a self-contained RIFF/WAVE byte stream with a `fmt `, `cue `,
/// `LIST adtl` (labl / note / ltxt) and a 4-sample `data` chunk.
fn build_wav_with_cue_adtl() -> Vec<u8> {
    let mut body = Vec::new();
    put_fourcc(&mut body, b"WAVE");

    // fmt chunk — 16 bytes WAVEFORMAT for PCM (Microsoft RIFF MCI §"WAVE
    // Format Chunk"): wFormatTag=1, wChannels=1, dwSamplesPerSec=8000,
    // dwAvgBytesPerSec=16000, wBlockAlign=2, wBitsPerSample=16.
    put_fourcc(&mut body, b"fmt ");
    put_u32(&mut body, 16);
    put_u16(&mut body, 1); // PCM
    put_u16(&mut body, 1); // mono
    put_u32(&mut body, 8_000); // sample rate
    put_u32(&mut body, 16_000); // byte rate
    put_u16(&mut body, 2); // block align
    put_u16(&mut body, 16); // bits per sample

    // cue chunk — two records: id=1 at sample 0, id=7 at sample 480.
    // Per the spec's "Examples of File Position Values" table for the
    // single-`data`-chunk case, dwChunkStart/dwBlockStart are zero and
    // only dwSampleOffset carries the position.
    put_fourcc(&mut body, b"cue ");
    let cue_size = 4 + 2 * 24;
    put_u32(&mut body, cue_size as u32);
    put_u32(&mut body, 2); // dwCuePoints
    push_cue_record(&mut body, 1, 0, b"data", 0, 0, 0);
    push_cue_record(&mut body, 7, 480, b"data", 0, 0, 480);

    // LIST adtl chunk carrying a labl for cue 1, a note for cue 7, and
    // an ltxt for cue 1.
    let mut adtl: Vec<u8> = Vec::new();
    adtl.extend_from_slice(b"adtl");

    // labl: dwName=1, ZSTR "intro"
    adtl.extend_from_slice(b"labl");
    let labl_text = b"intro\0";
    put_u32(&mut adtl, (4 + labl_text.len()) as u32);
    put_u32(&mut adtl, 1);
    adtl.extend_from_slice(labl_text);
    if labl_text.len() % 2 == 1 {
        adtl.push(0);
    }

    // note: dwName=7, ZSTR "outro-end"
    adtl.extend_from_slice(b"note");
    let note_text = b"outro-end\0";
    put_u32(&mut adtl, (4 + note_text.len()) as u32);
    put_u32(&mut adtl, 7);
    adtl.extend_from_slice(note_text);
    if note_text.len() % 2 == 1 {
        adtl.push(0);
    }

    // ltxt: dwName=1, dwSampleLength=480, dwPurpose='scrp', wCountry=0,
    // wLanguage=0, wDialect=0, wCodePage=0, ZSTR "caption"
    adtl.extend_from_slice(b"ltxt");
    let ltxt_text = b"caption\0";
    let ltxt_body_len = 20 + ltxt_text.len();
    put_u32(&mut adtl, ltxt_body_len as u32);
    put_u32(&mut adtl, 1); // dwName
    put_u32(&mut adtl, 480); // dwSampleLength
    put_fourcc(&mut adtl, b"scrp"); // dwPurpose
    put_u16(&mut adtl, 0);
    put_u16(&mut adtl, 0);
    put_u16(&mut adtl, 0);
    put_u16(&mut adtl, 0);
    adtl.extend_from_slice(ltxt_text);
    if ltxt_text.len() % 2 == 1 {
        adtl.push(0);
    }

    put_fourcc(&mut body, b"LIST");
    put_u32(&mut body, adtl.len() as u32);
    body.extend_from_slice(&adtl);

    // data chunk: 4 samples = 8 bytes mono S16.
    put_fourcc(&mut body, b"data");
    put_u32(&mut body, 8);
    body.extend_from_slice(&[0u8; 8]);

    // RIFF wrapper.
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    put_u32(&mut out, body.len() as u32);
    out.extend_from_slice(&body);
    out
}

#[test]
fn cue_and_adtl_keys_appear_in_metadata() {
    let bytes = build_wav_with_cue_adtl();
    let mut reg = ContainerRegistry::new();
    register_containers(&mut reg);
    let mut codecs = CodecRegistry::new();
    register_codecs(&mut codecs);
    let reader: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let demuxer = reg
        .open_demuxer("wav", reader, &codecs)
        .expect("demuxer opens");
    let metadata: std::collections::HashMap<String, String> =
        demuxer.metadata().iter().cloned().collect();

    // cue chunk surface
    assert_eq!(metadata.get("wav:cue.count").map(|s| s.as_str()), Some("2"));
    assert_eq!(
        metadata.get("wav:cue.1.position").map(|s| s.as_str()),
        Some("0"),
    );
    assert_eq!(
        metadata.get("wav:cue.1.fcc_chunk").map(|s| s.as_str()),
        Some("data"),
    );
    assert_eq!(
        metadata.get("wav:cue.1.sample_offset").map(|s| s.as_str()),
        Some("0"),
    );
    assert_eq!(
        metadata.get("wav:cue.7.position").map(|s| s.as_str()),
        Some("480"),
    );
    assert_eq!(
        metadata.get("wav:cue.7.sample_offset").map(|s| s.as_str()),
        Some("480"),
    );
    // For the single-`data`-chunk PCM case the spec says
    // dwChunkStart / dwBlockStart are both zero — the parser elides zero
    // values to keep the key set small.
    assert!(!metadata.contains_key("wav:cue.1.chunk_start"));
    assert!(!metadata.contains_key("wav:cue.1.block_start"));

    // LIST adtl surface
    assert_eq!(
        metadata.get("wav:cue.1.label").map(|s| s.as_str()),
        Some("intro"),
    );
    assert_eq!(
        metadata.get("wav:cue.7.note").map(|s| s.as_str()),
        Some("outro-end"),
    );
    assert_eq!(
        metadata.get("wav:cue.1.ltxt.length").map(|s| s.as_str()),
        Some("480"),
    );
    assert_eq!(
        metadata.get("wav:cue.1.ltxt.purpose").map(|s| s.as_str()),
        Some("scrp"),
    );
    assert_eq!(
        metadata.get("wav:cue.1.ltxt.text").map(|s| s.as_str()),
        Some("caption"),
    );
}

#[test]
fn missing_cue_chunk_leaves_metadata_untouched() {
    // Same shape minus the cue + adtl chunks: no `wav:cue.*` keys at all.
    let mut body = Vec::new();
    put_fourcc(&mut body, b"WAVE");
    put_fourcc(&mut body, b"fmt ");
    put_u32(&mut body, 16);
    put_u16(&mut body, 1);
    put_u16(&mut body, 1);
    put_u32(&mut body, 8_000);
    put_u32(&mut body, 16_000);
    put_u16(&mut body, 2);
    put_u16(&mut body, 16);
    put_fourcc(&mut body, b"data");
    put_u32(&mut body, 8);
    body.extend_from_slice(&[0u8; 8]);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    put_u32(&mut bytes, body.len() as u32);
    bytes.extend_from_slice(&body);

    let mut reg = ContainerRegistry::new();
    register_containers(&mut reg);
    let mut codecs = CodecRegistry::new();
    register_codecs(&mut codecs);
    let reader: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let demuxer = reg
        .open_demuxer("wav", reader, &codecs)
        .expect("demuxer opens");
    assert!(
        demuxer
            .metadata()
            .iter()
            .all(|(k, _)| !k.starts_with("wav:cue.")),
        "no cue metadata expected"
    );
}
