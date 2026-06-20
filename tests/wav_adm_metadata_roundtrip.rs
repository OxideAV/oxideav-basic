//! Integration test: a WAV carrying BOTH an Acidizer `acid` loop chunk
//! and a BW64/ADM `chna` channel-allocation chunk survives a full
//! public mux → demux round-trip — PCM payload byte-for-byte, the typed
//! `WavDemuxer::{acid,chna}()` accessors equal the muxed values, and the
//! ITU-R BS.2088-2 §8.1/§3 metadata (ADM reference kind + common/custom
//! definition scope) reaches the surfaced `wav:chna.*` keys.
//!
//! These chunk shapes come from the staged clean-room layouts
//! `docs/container/riff/acid-chunk.md` and
//! `docs/container/riff/metadata/bs2088-chna-chunk-layout.md`.

use std::collections::HashMap;
use std::io::Cursor;

use oxideav_basic::wav::{
    open_muxer_with, open_wav_demuxer, AcidChunk, AdmRefKind, AudioId, ChnaChunk, DefinitionScope,
    WavMuxOptions,
};
use oxideav_core::{
    CodecId, CodecParameters, Demuxer, Error, MediaType, Packet, ReadSeek, SampleFormat,
    StreamInfo, TimeBase, WriteSeek,
};

fn stereo_s16_stream(sample_rate: u32) -> StreamInfo {
    let mut params = CodecParameters::audio(CodecId::new("pcm_s16le"));
    params.media_type = MediaType::Audio;
    params.channels = Some(2);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
    StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, sample_rate as i64),
        duration: None,
        start_time: Some(0),
        params,
    }
}

/// Right-pad a string into a fixed-width NUL-padded char array the way an
/// ADM writer fills the `audioID` reference fields.
fn padded<const N: usize>(s: &str) -> [u8; N] {
    let mut out = [0u8; N];
    let b = s.as_bytes();
    out[..b.len()].copy_from_slice(b);
    out
}

fn audio_id(track_index: u16, uid: &str, track_ref: &str, pack_ref: &str) -> AudioId {
    AudioId {
        track_index,
        uid: padded::<12>(uid),
        track_ref: padded::<14>(track_ref),
        pack_ref: padded::<11>(pack_ref),
        pad: 0,
    }
}

#[test]
fn wav_with_acid_and_chna_round_trips() {
    let sample_rate = 48_000u32;

    // 0.05 s of distinct S16 stereo frames (interleaved L,R).
    let frames = sample_rate as usize / 20;
    let mut payload = Vec::with_capacity(frames * 4);
    for i in 0..frames {
        let l = (((i as i64) * 7).rem_euclid(65536) - 32768) as i16;
        let r = (((i as i64) * 11).rem_euclid(65536) - 32768) as i16;
        payload.extend_from_slice(&l.to_le_bytes());
        payload.extend_from_slice(&r.to_le_bytes());
    }

    // Acidizer loop metadata: a 4-beat stretched loop at 138 BPM with a
    // valid root note (flag bits: loop=0, root-set=0x02, stretch=0x04).
    let acid = AcidChunk {
        flags: 0x02 | 0x04,
        root_note: 60, // middle / High C
        reserved: [0; 6],
        num_beats: 4,
        meter: 4,
        tempo: 138.0,
    };

    // ADM channel allocation: a stereo pair where the two tracks
    // reference *common* BS.2094 channel definitions (trailing four hex
    // digits <= 0x0FFF) belonging to one *custom* pack (>= 0x1000). Track
    // refs use the `AC_` audioChannelFormat form (linear-PCM essence).
    let chna = ChnaChunk {
        num_tracks: 2,
        num_uids: 2,
        ids: vec![
            audio_id(1, "ATU_00000001", "AC_00010001_00", "AP_00011002"),
            audio_id(2, "ATU_00000002", "AC_00010002_00", "AP_00011002"),
        ],
    };

    // Mux through the public muxer with both chunks opted in.
    let opts = WavMuxOptions::default()
        .with_acid(acid)
        .with_chna(chna.clone());
    let stream = stereo_s16_stream(sample_rate);

    // `Box<dyn WriteSeek>` is `'static`, so a borrowed `Cursor<&mut Vec>`
    // can't satisfy it — mux to a uniquely-named tmpfile and read it back.
    let tmp = std::env::temp_dir().join("oxideav-basic-wav-adm-roundtrip.wav");
    let _ = std::fs::remove_file(&tmp);
    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = open_muxer_with(ws, std::slice::from_ref(&stream), opts).unwrap();
        mux.write_header().unwrap();
        let pkt = Packet::new(0, stream.time_base, payload.clone());
        mux.write_packet(&pkt).unwrap();
        mux.write_trailer().unwrap();
    }
    let buf = std::fs::read(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);

    // Both chunks must appear verbatim on the wire (header + body).
    let acid_body = acid.to_bytes();
    let mut acid_chunk = b"acid".to_vec();
    acid_chunk.extend_from_slice(&(acid_body.len() as u32).to_le_bytes());
    acid_chunk.extend_from_slice(&acid_body);
    assert!(
        buf.windows(acid_chunk.len()).any(|w| w == &acid_chunk[..]),
        "acid chunk must be present verbatim"
    );

    let chna_body = chna.to_bytes();
    let mut chna_chunk = b"chna".to_vec();
    chna_chunk.extend_from_slice(&(chna_body.len() as u32).to_le_bytes());
    chna_chunk.extend_from_slice(&chna_body);
    assert!(
        buf.windows(chna_chunk.len()).any(|w| w == &chna_chunk[..]),
        "chna chunk must be present verbatim"
    );

    // Demux and verify both typed accessors round-trip exactly.
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(buf));
    let mut dmx = open_wav_demuxer(rs).unwrap();

    let got_acid = dmx.acid().copied().expect("acid accessor populated");
    assert_eq!(got_acid, acid);
    assert!(!got_acid.one_shot());
    assert!(got_acid.root_note_set());
    assert!(got_acid.stretch());
    assert_eq!(got_acid.num_beats, 4);
    assert_eq!(got_acid.tempo, 138.0);

    assert_eq!(dmx.chna(), Some(&chna));
    let got_chna = dmx.chna().unwrap();
    assert_eq!(got_chna.defined_ids().count(), 2);

    // ADM reference classification (§1.2 / §3) on the typed records.
    let rec0 = &got_chna.ids[0];
    assert_eq!(rec0.track_ref_kind(), AdmRefKind::ChannelFormat);
    assert_eq!(rec0.track_ref_scope(), Some(DefinitionScope::Common));
    assert_eq!(rec0.pack_ref_kind(), AdmRefKind::PackFormat);
    assert_eq!(rec0.pack_ref_scope(), Some(DefinitionScope::Custom));

    // Surfaced metadata keys: acid + chna, including the new ADM
    // classification keys.
    let md: HashMap<String, String> = dmx.metadata().iter().cloned().collect();
    assert_eq!(md.get("wav:acid.tempo"), Some(&"138".to_string()));
    assert_eq!(md.get("wav:acid.stretch"), Some(&"1".to_string()));
    assert_eq!(md.get("wav:chna.num_tracks"), Some(&"2".to_string()));
    assert_eq!(md.get("wav:chna.defined_count"), Some(&"2".to_string()));
    assert_eq!(
        md.get("wav:chna.0.track_ref_kind"),
        Some(&"audioChannelFormat".to_string())
    );
    assert_eq!(
        md.get("wav:chna.0.track_ref_definition"),
        Some(&"common".to_string())
    );
    assert_eq!(
        md.get("wav:chna.0.pack_ref_definition"),
        Some(&"custom".to_string())
    );
    assert_eq!(
        md.get("wav:chna.1.track_ref_kind"),
        Some(&"audioChannelFormat".to_string())
    );

    // PCM payload survives byte-for-byte across the round-trip.
    let mut out = Vec::new();
    loop {
        match dmx.next_packet() {
            Ok(p) => out.extend_from_slice(&p.data),
            Err(Error::Eof) => break,
            Err(e) => panic!("demux error: {e}"),
        }
    }
    assert_eq!(out, payload, "PCM payload must round-trip byte-for-byte");
}
