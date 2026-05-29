//! Minimal pure-Rust RIFF/WAVE container.
//!
//! Supports reading and writing linear PCM streams via the `pcm_*` codecs
//! (`pcm_u8`, `pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_f32le`,
//! `pcm_f64le`) and dispatches `WAVE_FORMAT_ALAW (0x0006)` /
//! `WAVE_FORMAT_MULAW (0x0007)` streams to the external `pcm_alaw` /
//! `pcm_mulaw` codecs (provided by `oxideav-g711`, applied at decode
//! time by the host runtime — this crate's WAV layer only handles
//! framing).
//!
//! ## WAVEFORMATEXTENSIBLE (`wFormatTag = 0xFFFE`)
//!
//! Per `docs/container/riff/waveformatextensible/README.md` the demuxer
//! parses the 22-byte extension and surfaces:
//!
//! - `wValidBitsPerSample` — actual bit precision (may differ from the
//!   `wBitsPerSample` container size for 24-in-32-bit PCM).
//! - `dwChannelMask` — `SPEAKER_*` bitmap describing the channel
//!   ordering of the interleaved PCM byte stream.
//! - `SubFormat` GUID — the codec identifier when the legacy
//!   `wFormatTag` is the EXTENSIBLE escape hatch.
//!
//! Well-known `KSDATAFORMAT_SUBTYPE_*` GUIDs (PCM, IEEE_FLOAT, ALAW,
//! MULAW) are mapped to the same codec ids the legacy
//! `WAVEFORMATEX` path would have produced. Unknown GUIDs synthesise a
//! `wav:guid_<canonical-text>` codec id so downstream
//! `CodecRegistry::make_decoder` lookups fail cleanly naming the
//! actual GUID.
//!
//! The extension fields are also exposed verbatim through
//! `Demuxer::metadata` under the keys
//! `wav:fmt.valid_bits_per_sample` / `wav:fmt.channel_mask` /
//! `wav:fmt.subformat` (matching the round-75 `oxideav-avi` shape, but
//! single-stream so no per-stream index).

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, Error, MediaType, Packet, Result, SampleFormat,
    StreamInfo, TimeBase,
};
use oxideav_core::{ContainerRegistry, Demuxer, Muxer, ReadSeek, WriteSeek};
use std::io::{Read, Seek, SeekFrom, Write};

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("wav", open_demuxer);
    reg.register_muxer("wav", open_muxer);
    reg.register_extension("wav", "wav");
    reg.register_extension("wave", "wav");
    reg.register_probe("wav", probe);
}

/// `RIFF....WAVE` — unambiguous when present.
fn probe(p: &oxideav_core::ProbeData) -> u8 {
    if p.buf.len() < 12 {
        return 0;
    }
    if &p.buf[0..4] == b"RIFF" && &p.buf[8..12] == b"WAVE" {
        100
    } else {
        0
    }
}

// On-the-wire `wFormatTag` constants from RFC 2361 / `mmreg.h`. Public so
// muxer callers can build `WAVE_FORMAT_EXTENSIBLE` streams against the
// same dispatch table the demuxer uses.
/// `WAVE_FORMAT_PCM` — integer linear PCM (`mmreg.h`).
pub const WAVE_FORMAT_PCM: u16 = 0x0001;
/// `WAVE_FORMAT_IEEE_FLOAT` — 32-bit / 64-bit IEEE 754 float PCM.
pub const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
/// `WAVE_FORMAT_ALAW` — ITU-T G.711 A-law (RFC 2361 A.7).
pub const WAVE_FORMAT_ALAW: u16 = 0x0006;
/// `WAVE_FORMAT_MULAW` — ITU-T G.711 μ-law (RFC 2361 A.8).
pub const WAVE_FORMAT_MULAW: u16 = 0x0007;
/// `WAVE_FORMAT_EXTENSIBLE` — escape hatch with 22-byte extension
/// carrying `wValidBitsPerSample` / `dwChannelMask` / `SubFormat` GUID
/// (per docs/container/riff/waveformatextensible/README.md).
pub const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

// Internal aliases kept for readability of the local match arms below.
const FMT_PCM: u16 = WAVE_FORMAT_PCM;
const FMT_IEEE_FLOAT: u16 = WAVE_FORMAT_IEEE_FLOAT;
const FMT_ALAW: u16 = WAVE_FORMAT_ALAW;
const FMT_MULAW: u16 = WAVE_FORMAT_MULAW;
const FMT_EXTENSIBLE: u16 = WAVE_FORMAT_EXTENSIBLE;

// `KSDATAFORMAT_SUBTYPE_*` GUIDs (`KSMedia.h`). All follow the same
// `<tag>-0000-0010-8000-00AA00389B71` "DataFormat" base where the
// leading 16-bit `<tag>` is the legacy `wFormatTag` — per
// docs/container/riff/waveformatextensible/README.md.
const GUID_PCM: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
const GUID_IEEE_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
const GUID_ALAW: [u8; 16] = [
    0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
const GUID_MULAW: [u8; 16] = [
    0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// Format the 16-byte SubFormat GUID for diagnostic strings as
/// `XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX` (canonical text representation
/// used by `mmreg.h` GUID definitions). The first three groups are
/// little-endian on the wire, the trailing two groups are big-endian.
fn fmt_guid(g: &[u8; 16]) -> String {
    format!(
        "{:08X}-{:04X}-{:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
        u32::from_le_bytes([g[0], g[1], g[2], g[3]]),
        u16::from_le_bytes([g[4], g[5]]),
        u16::from_le_bytes([g[6], g[7]]),
        g[8],
        g[9],
        g[10],
        g[11],
        g[12],
        g[13],
        g[14],
        g[15],
    )
}

// --- Demuxer ---------------------------------------------------------------

fn open_demuxer(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> Result<Box<dyn Demuxer>> {
    let mut hdr = [0u8; 12];
    input.read_exact(&mut hdr)?;
    if &hdr[0..4] != b"RIFF" || &hdr[8..12] != b"WAVE" {
        return Err(Error::invalid("not a RIFF/WAVE file"));
    }

    // Walk chunks until we hit "data"; parse "fmt " and "LIST" along the way.
    let mut fmt: Option<WaveFmt> = None;
    let mut metadata: Vec<(String, String)> = Vec::new();
    let data_offset: u64;
    let data_size: u64;
    loop {
        let mut chdr = [0u8; 8];
        input.read_exact(&mut chdr)?;
        let id = &chdr[0..4];
        let size = u32::from_le_bytes([chdr[4], chdr[5], chdr[6], chdr[7]]) as u64;
        match id {
            b"fmt " => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                fmt = Some(parse_fmt(&buf)?);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"LIST" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_list_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"bext" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_bext_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"cue " => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_cue_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"smpl" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_smpl_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"inst" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_inst_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"data" => {
                data_offset = input.stream_position()?;
                data_size = size;
                break;
            }
            _ => {
                let pad = size + (size % 2);
                input.seek(SeekFrom::Current(pad as i64))?;
            }
        }
    }
    let fmt = fmt.ok_or_else(|| Error::invalid("WAV missing fmt chunk"))?;

    let codec_id = resolve_codec(&fmt)?;
    // Sample format hint for the decoded shape (NOT the on-wire layout
    // — A-law/μ-law expand to S16 once decoded, mirroring the
    // round-75 `oxideav-avi` audio_codec_sample_format helper). For
    // synthesised `wav:guid_<...>` ids the decoded shape is unknown
    // so we leave sample_format as None — callers can still pull
    // channels / sample_rate / channel_mask / SubFormat to surface a
    // useful diagnostic without us pretending to know the codec.
    let sample_fmt = decoded_sample_format(&codec_id);

    let time_base = TimeBase::new(1, fmt.sample_rate as i64);
    let block_align = fmt.block_align.max(1) as u64;
    let total_samples = data_size / block_align;
    let duration_micros: i64 = if fmt.sample_rate > 0 {
        (total_samples as i128 * 1_000_000 / fmt.sample_rate as i128) as i64
    } else {
        0
    };

    let mut params = CodecParameters::audio(codec_id);
    params.tag = Some(oxideav_core::CodecTag::wave_format(fmt.format_tag));
    params.channels = Some(fmt.channels);
    params.sample_rate = Some(fmt.sample_rate);
    params.sample_format = sample_fmt;
    // bit_rate uses the on-wire bytes_per_second (== block_align *
    // sample_rate) — for A-law/μ-law that's 8 * channels * rate, NOT
    // the post-decode S16 rate.
    params.bit_rate = Some(8 * block_align * (fmt.sample_rate as u64));

    // Round-77 metadata: surface WAVEFORMATEXTENSIBLE side-info under
    // the same key shape `oxideav-avi` uses (without the per-stream
    // index — WAV is single-stream by construction). Only emitted when
    // the on-wire wFormatTag is EXTENSIBLE and the extension parsed.
    if fmt.format_tag == FMT_EXTENSIBLE {
        if let Some(valid) = fmt.valid_bits_per_sample {
            metadata.push((
                "wav:fmt.valid_bits_per_sample".to_string(),
                valid.to_string(),
            ));
        }
        if let Some(mask) = fmt.channel_mask {
            metadata.push(("wav:fmt.channel_mask".to_string(), format!("0x{mask:08X}")));
        }
        if let Some(sub) = &fmt.subformat {
            metadata.push(("wav:fmt.subformat".to_string(), fmt_guid(sub)));
        }
    }

    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(total_samples as i64),
        start_time: Some(0),
        params,
    };

    Ok(Box::new(WavDemuxer {
        input,
        streams: vec![stream],
        data_offset,
        data_end: data_offset + data_size,
        cursor: data_offset,
        block_align,
        chunk_frames: 1024,
        samples_emitted: 0,
        metadata,
        duration_micros,
        format_tag: fmt.format_tag,
        valid_bits_per_sample: fmt.valid_bits_per_sample,
        channel_mask: fmt.channel_mask,
        subformat: fmt.subformat,
    }))
}

/// Parse a RIFF LIST chunk body. Dispatches by the 4-byte list type:
/// `INFO` maps `IART`/`INAM`/... sub-chunks to standard key names
/// (`artist`, `title`, ...); `adtl` (Associated Data List, per
/// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "Associated
/// Data Chunk") maps `labl`/`note`/`ltxt` sub-chunks to
/// `wav:adtl.<id>.<dwName>` keys carrying the cue-point text.
fn parse_list_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    if buf.len() < 4 {
        return;
    }
    match &buf[0..4] {
        b"INFO" => parse_info_list(&buf[4..], out),
        b"adtl" => parse_adtl_list(&buf[4..], out),
        _ => {}
    }
}

fn parse_info_list(buf: &[u8], out: &mut Vec<(String, String)>) {
    let mut i = 0usize;
    while i + 8 <= buf.len() {
        let id: [u8; 4] = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
        let size = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
        i += 8;
        if i + size > buf.len() {
            break;
        }
        let raw = &buf[i..i + size];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let value = String::from_utf8_lossy(&raw[..end]).trim().to_string();
        let key = info_id_to_key(&id);
        if !value.is_empty() {
            if let Some(k) = key {
                out.push((k.to_string(), value));
            }
        }
        i += size;
        if size % 2 == 1 {
            i += 1;
        }
    }
}

/// Parse a `LIST adtl` (Associated Data List) body and emit
/// `wav:adtl.<sub-id>.<dwName>` keys carrying the text payload of each
/// sub-chunk. Per `docs/container/riff/metadata/microsoft-riffmci.pdf`
/// §3 "Associated Data Chunk":
///
/// * `labl(<dwName:DWORD> <data:ZSTR>)` — title text for cue `dwName`.
/// * `note(<dwName:DWORD> <data:ZSTR>)` — comment text for cue `dwName`.
/// * `ltxt(<dwName> <dwSampleLength> <dwPurpose> <wCountry> <wLanguage>
///   <wDialect> <wCodePage> <data:BYTE>...)` — text covering a
///   `dwSampleLength`-sample segment starting at cue `dwName`. The
///   parser surfaces the segment length under `.ltxt.<dwName>.length`,
///   the FOURCC purpose under `.ltxt.<dwName>.purpose`, and the text
///   payload (trimmed at the first NUL) under `.ltxt.<dwName>.text`.
/// * `file(...)` — embedded media file; skipped (binary, no useful
///   metadata key without a reader for the inner format type).
///
/// Sub-chunks shorter than the minimum required header are skipped.
fn parse_adtl_list(buf: &[u8], out: &mut Vec<(String, String)>) {
    let mut i = 0usize;
    while i + 8 <= buf.len() {
        let id: [u8; 4] = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
        let size = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
        i += 8;
        if i + size > buf.len() {
            break;
        }
        let body = &buf[i..i + size];
        match &id {
            b"labl" | b"note" if body.len() >= 4 => {
                let dw_name = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);
                let raw = &body[4..];
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                let text = String::from_utf8_lossy(&raw[..end]).trim().to_string();
                if !text.is_empty() {
                    let sub = if &id == b"labl" { "labl" } else { "note" };
                    out.push((format!("wav:adtl.{sub}.{dw_name}"), text));
                }
            }
            // 4 dwName + 4 dwSampleLength + 4 dwPurpose + 2 wCountry
            // + 2 wLanguage + 2 wDialect + 2 wCodePage = 20 bytes
            // fixed header.
            b"ltxt" if body.len() >= 20 => {
                let dw_name = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);
                let dw_length = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                let purpose = &body[8..12];
                out.push((
                    format!("wav:adtl.ltxt.{dw_name}.length"),
                    dw_length.to_string(),
                ));
                // Render purpose as a FOURCC when printable, hex otherwise.
                let purpose_str = if purpose.iter().all(|&b| (0x20..=0x7E).contains(&b)) {
                    String::from_utf8_lossy(purpose).to_string()
                } else {
                    format!(
                        "0x{:02X}{:02X}{:02X}{:02X}",
                        purpose[0], purpose[1], purpose[2], purpose[3]
                    )
                };
                out.push((format!("wav:adtl.ltxt.{dw_name}.purpose"), purpose_str));
                let raw = &body[20..];
                // The text payload may or may not be NUL-terminated
                // per the spec ("<data:BYTE>..."); trim at the first
                // NUL if present and strip surrounding whitespace.
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                let text = String::from_utf8_lossy(&raw[..end]).trim().to_string();
                if !text.is_empty() {
                    out.push((format!("wav:adtl.ltxt.{dw_name}.text"), text));
                }
            }
            // `file` sub-chunks carry an embedded media file; we don't
            // surface their bytes through the string-typed metadata API.
            // Truncated `labl`/`note`/`ltxt` (under the fixed-header
            // minimum) are likewise skipped as opaque.
            _ => {}
        }
        i += size;
        if size % 2 == 1 {
            i += 1;
        }
    }
}

/// Parse a `cue ` chunk body and emit `wav:cue.count` + per-point
/// `wav:cue.<dwName>.position` / `.fcc_chunk` / `.chunk_start` /
/// `.block_start` / `.sample_offset` keys. Layout per
/// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "Cue-Points
/// Chunk":
///
/// ```text
/// <cue-ck> -> cue( <dwCuePoints:DWORD> <cue-point>... )
/// <cue-point> -> struct {
///     DWORD dwName;        // unique id, referenced by plst/adtl
///     DWORD dwPosition;    // sample position in the play order
///     FOURCC fccChunk;     // 'data' or 'slnt' (for wavl LIST forms)
///     DWORD dwChunkStart;  // byte offset of fccChunk within wavl LIST
///     DWORD dwBlockStart;  // byte offset of enclosing block
///     DWORD dwSampleOffset;// sample offset within block
/// }
/// ```
///
/// A truncated chunk (count > body) is treated as opaque and skipped.
/// Each cue-point record is 24 bytes; the function consumes as many
/// records as the body actually carries even if `dwCuePoints` claims
/// more (defensive vs. writers that lie about the count).
fn parse_cue_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    if buf.len() < 4 {
        return;
    }
    let count_claimed = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let body = &buf[4..];
    const REC_LEN: usize = 24;
    let count_actual = (body.len() / REC_LEN) as u32;
    let count = count_claimed.min(count_actual);
    out.push(("wav:cue.count".to_string(), count.to_string()));
    for i in 0..count as usize {
        let off = i * REC_LEN;
        let dw_name = u32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]);
        let dw_position =
            u32::from_le_bytes([body[off + 4], body[off + 5], body[off + 6], body[off + 7]]);
        let fcc_chunk = &body[off + 8..off + 12];
        let dw_chunk_start = u32::from_le_bytes([
            body[off + 12],
            body[off + 13],
            body[off + 14],
            body[off + 15],
        ]);
        let dw_block_start = u32::from_le_bytes([
            body[off + 16],
            body[off + 17],
            body[off + 18],
            body[off + 19],
        ]);
        let dw_sample_offset = u32::from_le_bytes([
            body[off + 20],
            body[off + 21],
            body[off + 22],
            body[off + 23],
        ]);
        let fcc_str = if fcc_chunk.iter().all(|&b| (0x20..=0x7E).contains(&b)) {
            String::from_utf8_lossy(fcc_chunk).to_string()
        } else {
            format!(
                "0x{:02X}{:02X}{:02X}{:02X}",
                fcc_chunk[0], fcc_chunk[1], fcc_chunk[2], fcc_chunk[3]
            )
        };
        out.push((
            format!("wav:cue.{dw_name}.position"),
            dw_position.to_string(),
        ));
        out.push((format!("wav:cue.{dw_name}.fcc_chunk"), fcc_str));
        out.push((
            format!("wav:cue.{dw_name}.chunk_start"),
            dw_chunk_start.to_string(),
        ));
        out.push((
            format!("wav:cue.{dw_name}.block_start"),
            dw_block_start.to_string(),
        ));
        out.push((
            format!("wav:cue.{dw_name}.sample_offset"),
            dw_sample_offset.to_string(),
        ));
    }
}

/// Parse a `smpl` (Sampler) chunk body and emit `wav:smpl.*` metadata
/// keys. Layout per the RIFF MCI / `mmreg.h`-era Sampler structure as
/// catalogued in
/// `docs/container/riff/metadata/exiftool-riff-tags.html` § "RIFF
/// Sampler Tags" and summarised in
/// `docs/container/riff/metadata/README.md` § "Sampler / Instrument
/// chunks":
///
/// ```text
/// <smpl-ck> -> smpl( <fixed:36 bytes> <loops:N × 24 bytes> <sampler-data> )
/// <fixed> -> struct {
///     DWORD dwManufacturer;
///     DWORD dwProduct;
///     DWORD dwSamplePeriod;       // nanoseconds per sample
///     DWORD dwMIDIUnityNote;      // 0..=127
///     DWORD dwMIDIPitchFraction;  // fractional offset, /0xFFFFFFFF
///     DWORD dwSMPTEFormat;        // 0/24/25/29/30 fps
///     DWORD dwSMPTEOffset;        // packed HH MM SS FF (MSB → LSB)
///     DWORD cSampleLoops;
///     DWORD cbSamplerData;        // bytes of trailing sampler-specific data
/// }
/// <loop> -> struct {
///     DWORD dwCuePointID;
///     DWORD dwType;               // 0=forward, 1=ping-pong, 2=reverse
///     DWORD dwStart;              // start sample offset
///     DWORD dwEnd;                // end sample offset
///     DWORD dwFraction;           // /0xFFFFFFFF fractional sample
///     DWORD dwPlayCount;          // 0 == infinite
/// }
/// ```
///
/// A `cSampleLoops` count exceeding what the chunk body actually
/// carries is clamped to the records that fit (defensive against
/// writers that lie about the count); a body shorter than the 36-byte
/// fixed header is treated as opaque and skipped.
fn parse_smpl_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    const FIXED_LEN: usize = 36;
    const LOOP_LEN: usize = 24;
    if buf.len() < FIXED_LEN {
        return;
    }
    let r = |off: usize| -> u32 {
        u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    };
    let manufacturer = r(0);
    let product = r(4);
    let sample_period = r(8);
    let midi_unity_note = r(12);
    let midi_pitch_fraction = r(16);
    let smpte_format = r(20);
    let smpte_offset = r(24);
    let num_loops_claimed = r(28);
    let sampler_data_len = r(32);

    out.push((
        "wav:smpl.manufacturer".to_string(),
        manufacturer.to_string(),
    ));
    out.push(("wav:smpl.product".to_string(), product.to_string()));
    out.push((
        "wav:smpl.sample_period".to_string(),
        sample_period.to_string(),
    ));
    out.push((
        "wav:smpl.midi_unity_note".to_string(),
        midi_unity_note.to_string(),
    ));
    out.push((
        "wav:smpl.midi_pitch_fraction".to_string(),
        midi_pitch_fraction.to_string(),
    ));
    out.push((
        "wav:smpl.smpte_format".to_string(),
        smpte_format.to_string(),
    ));
    // SMPTE offset packs HH MM SS FF in the high-to-low bytes of the
    // DWORD. Render the canonical HH:MM:SS:FF form alongside the raw
    // value for callers that prefer the on-wire integer.
    let smpte_hh = (smpte_offset >> 24) & 0xFF;
    let smpte_mm = (smpte_offset >> 16) & 0xFF;
    let smpte_ss = (smpte_offset >> 8) & 0xFF;
    let smpte_ff = smpte_offset & 0xFF;
    out.push((
        "wav:smpl.smpte_offset".to_string(),
        format!("{smpte_hh:02}:{smpte_mm:02}:{smpte_ss:02}:{smpte_ff:02}"),
    ));
    out.push((
        "wav:smpl.sampler_data_len".to_string(),
        sampler_data_len.to_string(),
    ));

    let body = &buf[FIXED_LEN..];
    let num_loops_fits = (body.len() / LOOP_LEN) as u32;
    let num_loops = num_loops_claimed.min(num_loops_fits);
    out.push((
        "wav:smpl.num_sample_loops".to_string(),
        num_loops.to_string(),
    ));
    for i in 0..num_loops as usize {
        let off = i * LOOP_LEN;
        let loop_field = |word: usize| -> u32 {
            let p = off + word * 4;
            u32::from_le_bytes([body[p], body[p + 1], body[p + 2], body[p + 3]])
        };
        let cue_point_id = loop_field(0);
        let loop_type = loop_field(1);
        let start = loop_field(2);
        let end = loop_field(3);
        let fraction = loop_field(4);
        let play_count = loop_field(5);
        out.push((
            format!("wav:smpl.loop.{i}.cue_point_id"),
            cue_point_id.to_string(),
        ));
        out.push((format!("wav:smpl.loop.{i}.type"), loop_type.to_string()));
        out.push((format!("wav:smpl.loop.{i}.start"), start.to_string()));
        out.push((format!("wav:smpl.loop.{i}.end"), end.to_string()));
        out.push((format!("wav:smpl.loop.{i}.fraction"), fraction.to_string()));
        out.push((
            format!("wav:smpl.loop.{i}.play_count"),
            play_count.to_string(),
        ));
    }
}

/// Parse an `inst` (Instrument) chunk body and emit `wav:inst.*`
/// metadata keys. Layout per
/// `docs/container/riff/metadata/exiftool-riff-tags.html` § "RIFF
/// Instrument Tags":
///
/// ```text
/// <inst-ck> -> inst( <UnshiftedNote:i8> <FineTune:i8> <Gain:i8>
///                    <LowNote:u8> <HighNote:u8>
///                    <LowVelocity:u8> <HighVelocity:u8> )
/// ```
///
/// `UnshiftedNote`, `LowNote`, `HighNote` are MIDI note numbers
/// (0..=127); `FineTune` is cents and `Gain` is dB, both signed
/// 8-bit. Velocity range is 1..=127 (unsigned).
///
/// A body shorter than the 7-byte fixed struct is treated as opaque
/// and skipped.
fn parse_inst_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    const FIXED_LEN: usize = 7;
    if buf.len() < FIXED_LEN {
        return;
    }
    let unshifted_note = buf[0];
    let fine_tune = buf[1] as i8;
    let gain = buf[2] as i8;
    let low_note = buf[3];
    let high_note = buf[4];
    let low_velocity = buf[5];
    let high_velocity = buf[6];
    out.push((
        "wav:inst.unshifted_note".to_string(),
        unshifted_note.to_string(),
    ));
    out.push(("wav:inst.fine_tune".to_string(), fine_tune.to_string()));
    out.push(("wav:inst.gain".to_string(), gain.to_string()));
    out.push(("wav:inst.low_note".to_string(), low_note.to_string()));
    out.push(("wav:inst.high_note".to_string(), high_note.to_string()));
    out.push((
        "wav:inst.low_velocity".to_string(),
        low_velocity.to_string(),
    ));
    out.push((
        "wav:inst.high_velocity".to_string(),
        high_velocity.to_string(),
    ));
}

fn info_id_to_key(id: &[u8; 4]) -> Option<&'static str> {
    match id {
        b"INAM" => Some("title"),
        b"IART" => Some("artist"),
        b"IPRD" => Some("album"),
        b"ICMT" => Some("comment"),
        b"ICRD" => Some("date"),
        b"IGNR" => Some("genre"),
        b"ICOP" => Some("copyright"),
        b"IENG" => Some("engineer"),
        b"ITCH" => Some("technician"),
        b"ISFT" => Some("encoder"),
        b"ISBJ" => Some("subject"),
        b"ITRK" => Some("track"),
        _ => None,
    }
}

/// Fixed (pre-`CodingHistory`) size of the BWF `bext` struct, in bytes.
///
/// Sum of the field widths from EBU Tech 3285 v2 §2.3 `BROADCAST_EXT`:
/// `Description[256] + Originator[32] + OriginatorReference[32] +
/// OriginationDate[10] + OriginationTime[8] + TimeReferenceLow(4) +
/// TimeReferenceHigh(4) + Version(2) + UMID[64] + LoudnessValue(2) +
/// LoudnessRange(2) + MaxTruePeakLevel(2) + MaxMomentaryLoudness(2) +
/// MaxShortTermLoudness(2) + Reserved[180]` = 602.
const BEXT_FIXED_LEN: usize = 602;

/// Trim a fixed-width ASCII field to its value: cut at the first NUL
/// (EBU Tech 3285 v2 §2.3 mandates a NUL terminator for under-length
/// strings) and strip surrounding whitespace.
fn bext_field(raw: &[u8]) -> String {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).trim().to_string()
}

/// Parse a BWF `bext` (Broadcast Audio Extension) chunk body and append
/// its fields to `out` under `wav:bext.*` keys.
///
/// Layout per `docs/container/riff/metadata/ebu-tech3285-bwf.pdf`
/// (EBU Tech 3285 v2 §2.3, `BROADCAST_EXT` struct). All multi-byte
/// integers are little-endian (RIFF convention). The loudness fields
/// (`LoudnessValue` … `MaxShortTermLoudness`) are signed 16-bit values
/// equal to `round(100 × value)` per §2.4, so they are surfaced divided
/// by 100 with two decimal places.
///
/// The `Version` field selects which fields are meaningful: v0 has none
/// of the UMID/loudness fields populated, v1 adds the SMPTE-330M UMID,
/// v2 adds the five loudness values (§1.1). The fixed struct is always
/// 602 bytes regardless of version, so this parser reads every field
/// unconditionally and lets the version key tell the caller which ones
/// to trust. `TimeReference` is reassembled as a 64-bit sample count.
fn parse_bext_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    if buf.len() < BEXT_FIXED_LEN {
        return;
    }
    let description = bext_field(&buf[0..256]);
    let originator = bext_field(&buf[256..288]);
    let originator_ref = bext_field(&buf[288..320]);
    let origination_date = bext_field(&buf[320..330]);
    let origination_time = bext_field(&buf[330..338]);
    let time_ref_low = u32::from_le_bytes([buf[338], buf[339], buf[340], buf[341]]);
    let time_ref_high = u32::from_le_bytes([buf[342], buf[343], buf[344], buf[345]]);
    let time_reference = ((time_ref_high as u64) << 32) | (time_ref_low as u64);
    let version = u16::from_le_bytes([buf[346], buf[347]]);

    // UMID[64] at [348..412]; only meaningful for v1+. Surface it as a
    // hex string only when non-zero so v0 files don't carry a bogus
    // all-zero UMID key.
    let umid = &buf[348..412];

    // Loudness WORDs at [412..422]; signed, ×100. Only meaningful for v2.
    let loudness_value = i16::from_le_bytes([buf[412], buf[413]]);
    let loudness_range = i16::from_le_bytes([buf[414], buf[415]]);
    let max_true_peak = i16::from_le_bytes([buf[416], buf[417]]);
    let max_momentary = i16::from_le_bytes([buf[418], buf[419]]);
    let max_short_term = i16::from_le_bytes([buf[420], buf[421]]);

    // CodingHistory: everything past the 602-byte fixed struct.
    let coding_history = if buf.len() > BEXT_FIXED_LEN {
        bext_field(&buf[BEXT_FIXED_LEN..])
    } else {
        String::new()
    };

    let push = |out: &mut Vec<(String, String)>, key: &str, value: String| {
        if !value.is_empty() {
            out.push((key.to_string(), value));
        }
    };

    push(out, "wav:bext.description", description);
    push(out, "wav:bext.originator", originator);
    push(out, "wav:bext.originator_reference", originator_ref);
    push(out, "wav:bext.origination_date", origination_date);
    push(out, "wav:bext.origination_time", origination_time);
    out.push((
        "wav:bext.time_reference".to_string(),
        time_reference.to_string(),
    ));
    out.push(("wav:bext.version".to_string(), version.to_string()));

    // v1+ : the SMPTE-330M UMID (64 bytes; a 32-byte "basic UMID"
    // zero-pads the trailing half per §2.3). Emit only when present.
    if umid.iter().any(|&b| b != 0) {
        let mut hex = String::with_capacity(umid.len() * 2);
        for b in umid {
            hex.push_str(&format!("{b:02x}"));
        }
        out.push(("wav:bext.umid".to_string(), hex));
    }

    // v2 : loudness metadata (×100 fixed-point → two decimals). Emitted
    // only for v2 files since v0/v1 leave these WORDs zero by spec.
    if version >= 2 {
        out.push((
            "wav:bext.loudness_value".to_string(),
            fmt_loudness(loudness_value),
        ));
        out.push((
            "wav:bext.loudness_range".to_string(),
            fmt_loudness(loudness_range),
        ));
        out.push((
            "wav:bext.max_true_peak_level".to_string(),
            fmt_loudness(max_true_peak),
        ));
        out.push((
            "wav:bext.max_momentary_loudness".to_string(),
            fmt_loudness(max_momentary),
        ));
        out.push((
            "wav:bext.max_short_term_loudness".to_string(),
            fmt_loudness(max_short_term),
        ));
    }

    push(out, "wav:bext.coding_history", coding_history);
}

/// Render a BWF loudness WORD (signed 16-bit, `round(100 × value)` per
/// EBU Tech 3285 v2 §2.4) back to its two-decimal value, e.g. `-2264`
/// → `"-22.64"`. The sign is carried on the integer part so values in
/// `(-1, 0)` keep their leading minus (`-50` → `"-0.50"`).
fn fmt_loudness(v: i16) -> String {
    let neg = v < 0;
    let abs = (v as i32).unsigned_abs();
    let whole = abs / 100;
    let frac = abs % 100;
    if neg {
        format!("-{whole}.{frac:02}")
    } else {
        format!("{whole}.{frac:02}")
    }
}

#[derive(Clone, Debug)]
struct WaveFmt {
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    #[allow(dead_code)]
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    /// `wValidBitsPerSample` from the 22-byte EXTENSIBLE extension.
    /// `None` for plain `WAVEFORMATEX`; `Some(0)` is a writer that left
    /// the union zero — the demuxer falls back to `bits_per_sample`.
    valid_bits_per_sample: Option<u16>,
    /// `dwChannelMask` — SPEAKER_* bitmap. `None` outside EXTENSIBLE.
    channel_mask: Option<u32>,
    /// 16-byte SubFormat GUID. `None` outside EXTENSIBLE.
    subformat: Option<[u8; 16]>,
}

fn parse_fmt(buf: &[u8]) -> Result<WaveFmt> {
    if buf.len() < 16 {
        return Err(Error::invalid("fmt chunk too small"));
    }
    let format_tag = u16::from_le_bytes([buf[0], buf[1]]);
    let channels = u16::from_le_bytes([buf[2], buf[3]]);
    let sample_rate = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let byte_rate = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let block_align = u16::from_le_bytes([buf[12], buf[13]]);
    let bits_per_sample = u16::from_le_bytes([buf[14], buf[15]]);
    let mut valid_bits_per_sample = None;
    let mut channel_mask = None;
    let mut subformat = None;
    if format_tag == FMT_EXTENSIBLE {
        // WAVEFORMATEXTENSIBLE layout per
        // docs/container/riff/waveformatextensible/README.md
        // §"Structure layout":
        //   [16..18]  cbSize             (must be >= 22)
        //   [18..20]  wValidBitsPerSample (active union member)
        //   [20..24]  dwChannelMask
        //   [24..40]  SubFormat GUID
        if buf.len() < 40 {
            return Err(Error::invalid(
                "WAVE_FORMAT_EXTENSIBLE fmt chunk shorter than the 40 bytes \
                 mandated by mmreg.h",
            ));
        }
        let cb_size = u16::from_le_bytes([buf[16], buf[17]]);
        if (cb_size as usize) < 22 {
            return Err(Error::invalid(format!(
                "WAVE_FORMAT_EXTENSIBLE cbSize must be >= 22, got {cb_size}"
            )));
        }
        valid_bits_per_sample = Some(u16::from_le_bytes([buf[18], buf[19]]));
        channel_mask = Some(u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]));
        let mut g = [0u8; 16];
        g.copy_from_slice(&buf[24..40]);
        subformat = Some(g);
    }
    Ok(WaveFmt {
        format_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        valid_bits_per_sample,
        channel_mask,
        subformat,
    })
}

fn resolve_codec(fmt: &WaveFmt) -> Result<CodecId> {
    match fmt.format_tag {
        FMT_PCM => Ok(CodecId::new(pcm_int_codec(fmt.bits_per_sample)?)),
        FMT_IEEE_FLOAT => Ok(CodecId::new(pcm_float_codec(fmt.bits_per_sample)?)),
        FMT_ALAW => Ok(CodecId::new("pcm_alaw")),
        FMT_MULAW => Ok(CodecId::new("pcm_mulaw")),
        FMT_EXTENSIBLE => {
            let sub = fmt
                .subformat
                .ok_or_else(|| Error::invalid("extensible WAV missing subformat"))?;
            // Per docs/container/riff/waveformatextensible/README.md
            // §"On using these specs": the actual codec precision is the
            // SubFormat union's wValidBitsPerSample, NOT the WAVEFORMATEX
            // wBitsPerSample (the container size). Fall back to the
            // container size only when the union is zero — some writers
            // leave it unset.
            let depth = fmt
                .valid_bits_per_sample
                .filter(|&v| v > 0)
                .unwrap_or(fmt.bits_per_sample);
            match sub {
                GUID_PCM => Ok(CodecId::new(pcm_int_codec(depth)?)),
                GUID_IEEE_FLOAT => Ok(CodecId::new(pcm_float_codec(depth)?)),
                GUID_ALAW => Ok(CodecId::new("pcm_alaw")),
                GUID_MULAW => Ok(CodecId::new("pcm_mulaw")),
                // Unknown SubFormat — synthesise a `wav:guid_<text>` id
                // so downstream make_decoder fails naming the actual
                // GUID rather than the opaque 0xFFFE tag. Mirrors the
                // `avi:guid_<...>` pattern in oxideav-avi.
                other => Ok(CodecId::new(format!("wav:guid_{}", fmt_guid(&other)))),
            }
        }
        other => Err(Error::unsupported(format!(
            "unsupported WAV format tag 0x{:04x}",
            other
        ))),
    }
}

fn pcm_int_codec(bits: u16) -> Result<&'static str> {
    Ok(match bits {
        8 => "pcm_u8",
        16 => "pcm_s16le",
        24 => "pcm_s24le",
        32 => "pcm_s32le",
        _ => {
            return Err(Error::unsupported(format!(
                "unsupported WAV integer-PCM bit depth: {bits}"
            )));
        }
    })
}

fn pcm_float_codec(bits: u16) -> Result<&'static str> {
    Ok(match bits {
        32 => "pcm_f32le",
        64 => "pcm_f64le",
        _ => {
            return Err(Error::unsupported(format!(
                "unsupported WAV IEEE-float bit depth: {bits}"
            )));
        }
    })
}

/// Decoded sample-format hint for a WAV codec id. The host runtime
/// applies the actual decode (A-law/μ-law through `oxideav-g711`); this
/// crate only resolves what the *output* of that decode looks like in
/// the standard `SampleFormat` enum so callers building pipelines know
/// the shape ahead of time.
///
/// For unknown `wav:guid_<...>` ids the function returns `None` — the
/// EXTENSIBLE GUID didn't match any of the well-known SubFormats and
/// we don't pretend to know how the codec is decoded.
fn decoded_sample_format(id: &CodecId) -> Option<SampleFormat> {
    // Plain PCM codec ids forward to the existing helper.
    if let Some(fmt) = super::pcm::sample_format_for(id) {
        return Some(fmt);
    }
    match id.as_str() {
        // G.711 expands to S16 once decoded (oxideav-g711 alaw/mulaw
        // output is S16, matching the round-75 audio_codec_sample_format
        // mapping in oxideav-avi).
        "pcm_alaw" | "pcm_mulaw" => Some(SampleFormat::S16),
        _ => None,
    }
}

/// WAV demuxer.
///
/// Beyond the `Demuxer` trait, this type exposes round-77 accessors
/// for the `WAVEFORMATEXTENSIBLE` side-info — `format_tag`,
/// `valid_bits_per_sample`, `channel_mask`, `subformat`. Callers that
/// only have a `Box<dyn Demuxer>` should rely on the `wav:fmt.*`
/// metadata keys instead.
pub struct WavDemuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    data_offset: u64,
    data_end: u64,
    cursor: u64,
    block_align: u64,
    chunk_frames: u64,
    samples_emitted: i64,
    metadata: Vec<(String, String)>,
    duration_micros: i64,
    format_tag: u16,
    valid_bits_per_sample: Option<u16>,
    channel_mask: Option<u32>,
    subformat: Option<[u8; 16]>,
}

impl WavDemuxer {
    /// On-wire `wFormatTag` from the `fmt ` chunk (one of `WAVE_FORMAT_*`).
    /// Preserved verbatim for round-trip purposes — the codec id
    /// already encodes the decoder dispatch.
    pub fn format_tag(&self) -> u16 {
        self.format_tag
    }

    /// `WAVEFORMATEXTENSIBLE.Samples.wValidBitsPerSample` — actual bit
    /// precision per sample. `None` for legacy `WAVEFORMATEX` streams
    /// (non-EXTENSIBLE `wFormatTag`).
    pub fn valid_bits_per_sample(&self) -> Option<u16> {
        self.valid_bits_per_sample
    }

    /// `WAVEFORMATEXTENSIBLE.dwChannelMask` — `SPEAKER_*` bitmap
    /// describing the channel ordering of the interleaved PCM byte
    /// stream. `None` for non-EXTENSIBLE streams.
    ///
    /// See `docs/container/riff/waveformatextensible/README.md`
    /// §"dwChannelMask bits" for the standard layouts.
    pub fn channel_mask(&self) -> Option<u32> {
        self.channel_mask
    }

    /// `WAVEFORMATEXTENSIBLE.SubFormat` — 16-byte GUID (the actual
    /// codec identifier when `format_tag == WAVE_FORMAT_EXTENSIBLE`).
    /// Returned in on-wire byte order (first three groups
    /// little-endian, trailing two groups big-endian); use
    /// [`Self::subformat_text`] for the canonical text representation.
    pub fn subformat(&self) -> Option<&[u8; 16]> {
        self.subformat.as_ref()
    }

    /// `WAVEFORMATEXTENSIBLE.SubFormat` formatted as a canonical
    /// `XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX` GUID string.
    pub fn subformat_text(&self) -> Option<String> {
        self.subformat.as_ref().map(fmt_guid)
    }
}

impl Demuxer for WavDemuxer {
    fn format_name(&self) -> &str {
        "wav"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.cursor >= self.data_end {
            return Err(Error::Eof);
        }
        let remaining = self.data_end - self.cursor;
        let want_bytes = (self.chunk_frames * self.block_align).min(remaining);
        let want_bytes = (want_bytes / self.block_align) * self.block_align;
        if want_bytes == 0 {
            return Err(Error::Eof);
        }

        // Ensure we're positioned correctly (if an upstream operation seeked us).
        self.input.seek(SeekFrom::Start(self.cursor))?;
        let mut buf = vec![0u8; want_bytes as usize];
        self.input.read_exact(&mut buf)?;
        self.cursor += want_bytes;

        let stream = &self.streams[0];
        let frames = want_bytes / self.block_align;
        let pts = self.samples_emitted;
        self.samples_emitted += frames as i64;

        let mut pkt = Packet::new(0, stream.time_base, buf);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(frames as i64);
        pkt.flags.keyframe = true;
        Ok(pkt)
    }

    fn seek_to(&mut self, stream_index: u32, pts: i64) -> Result<i64> {
        if stream_index != 0 {
            return Err(Error::invalid(format!(
                "WAV: stream index {stream_index} out of range"
            )));
        }
        // PCM is keyframe-only and frame-aligned: the target pts is a
        // sample-index offset into the data chunk. Clamp to the valid
        // range and translate to a byte offset.
        let total_samples = (self.data_end - self.data_offset) / self.block_align;
        let target = (pts.max(0) as u64).min(total_samples);
        let new_cursor = self.data_offset + target * self.block_align;

        self.input.seek(SeekFrom::Start(new_cursor))?;
        self.cursor = new_cursor;
        self.samples_emitted = target as i64;
        Ok(target as i64)
    }

    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    fn duration_micros(&self) -> Option<i64> {
        if self.duration_micros > 0 {
            Some(self.duration_micros)
        } else {
            None
        }
    }
}

// --- Muxer -----------------------------------------------------------------

fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::unsupported("WAV supports exactly one audio stream"));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Audio {
        return Err(Error::invalid("WAV stream must be audio"));
    }
    let channels = s
        .params
        .channels
        .ok_or_else(|| Error::invalid("WAV muxer: missing channels"))?;
    let sample_rate = s
        .params
        .sample_rate
        .ok_or_else(|| Error::invalid("WAV muxer: missing sample rate"))?;
    // Codec-id directs which `wFormatTag` flavour and on-wire shape the
    // muxer writes. A-law / μ-law take the dedicated tag-6/7 paths;
    // every other id falls back to the PCM/IEEE-FLOAT sample-format
    // lookup. Extensible muxing is opt-in via [`WavMuxOptions`] (see
    // [`open_muxer_with`] below) — the default path writes the
    // legacy 16-byte `WAVEFORMAT` for maximum compatibility.
    let shape = wire_shape_for_params(&s.params)?;
    Ok(Box::new(WavMuxer {
        output,
        channels,
        sample_rate,
        shape,
        extensible: None,
        riff_size_offset: 0,
        data_size_offset: 0,
        data_bytes: 0,
        header_written: false,
        trailer_written: false,
    }))
}

/// Optional muxer configuration: write a `WAVE_FORMAT_EXTENSIBLE`
/// (`wFormatTag = 0xFFFE`) header with the supplied `dwChannelMask`,
/// `wValidBitsPerSample`, and SubFormat GUID. See
/// `docs/container/riff/waveformatextensible/README.md` §"Channel-mask"
/// for the standard layouts.
///
/// When `valid_bits_per_sample` is `None` the muxer reuses the
/// container `wBitsPerSample` (computed from the codec's
/// `SampleFormat`). When `subformat` is `None` the muxer picks the
/// well-known `KSDATAFORMAT_SUBTYPE_*` GUID for the codec id (PCM /
/// IEEE_FLOAT / ALAW / MULAW).
#[derive(Clone, Debug, Default)]
pub struct WavMuxOptions {
    extensible: Option<ExtensibleOpts>,
}

#[derive(Clone, Debug)]
struct ExtensibleOpts {
    channel_mask: u32,
    valid_bits_per_sample: Option<u16>,
    subformat: Option<[u8; 16]>,
}

impl WavMuxOptions {
    /// Opt into `WAVE_FORMAT_EXTENSIBLE` muxing with the supplied
    /// `dwChannelMask`. The muxer derives `wValidBitsPerSample` and
    /// SubFormat from the codec-id automatically; use the
    /// finer-grained setters to override.
    pub fn with_extensible(mut self, channel_mask: u32) -> Self {
        self.extensible = Some(ExtensibleOpts {
            channel_mask,
            valid_bits_per_sample: None,
            subformat: None,
        });
        self
    }

    /// Override `wValidBitsPerSample` for an extensible stream. Has no
    /// effect unless [`Self::with_extensible`] was also called.
    pub fn with_valid_bits_per_sample(mut self, valid_bps: u16) -> Self {
        if let Some(opts) = self.extensible.as_mut() {
            opts.valid_bits_per_sample = Some(valid_bps);
        }
        self
    }

    /// Override the 16-byte SubFormat GUID. Has no effect unless
    /// [`Self::with_extensible`] was also called.
    pub fn with_subformat(mut self, guid: [u8; 16]) -> Self {
        if let Some(opts) = self.extensible.as_mut() {
            opts.subformat = Some(guid);
        }
        self
    }
}

/// Open the WAV muxer with caller-controlled `WAVEFORMATEXTENSIBLE`
/// options. Identical to `open_muxer` when `opts ==
/// WavMuxOptions::default()`.
pub fn open_muxer_with(
    output: Box<dyn WriteSeek>,
    streams: &[StreamInfo],
    opts: WavMuxOptions,
) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::unsupported("WAV supports exactly one audio stream"));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Audio {
        return Err(Error::invalid("WAV stream must be audio"));
    }
    let channels = s
        .params
        .channels
        .ok_or_else(|| Error::invalid("WAV muxer: missing channels"))?;
    let sample_rate = s
        .params
        .sample_rate
        .ok_or_else(|| Error::invalid("WAV muxer: missing sample rate"))?;
    let shape = wire_shape_for_params(&s.params)?;
    Ok(Box::new(WavMuxer {
        output,
        channels,
        sample_rate,
        shape,
        extensible: opts.extensible,
        riff_size_offset: 0,
        data_size_offset: 0,
        data_bytes: 0,
        header_written: false,
        trailer_written: false,
    }))
}

/// On-wire shape the muxer needs to know per codec — distinguishes
/// PCM/IEEE-float (bit-depth driven) from A-law/μ-law (fixed
/// 8 bits / sample).
#[derive(Clone, Copy, Debug)]
enum WireShape {
    /// Integer PCM (`wFormatTag = 0x0001`). bits = `wBitsPerSample`.
    IntPcm { bits: u16 },
    /// IEEE float PCM (`wFormatTag = 0x0003`). bits = 32 or 64.
    FloatPcm { bits: u16 },
    /// G.711 A-law (`wFormatTag = 0x0006`), 8 bits per sample.
    Alaw,
    /// G.711 μ-law (`wFormatTag = 0x0007`), 8 bits per sample.
    Mulaw,
}

impl WireShape {
    fn bits_per_sample(self) -> u16 {
        match self {
            WireShape::IntPcm { bits } | WireShape::FloatPcm { bits } => bits,
            WireShape::Alaw | WireShape::Mulaw => 8,
        }
    }

    fn format_tag(self) -> u16 {
        match self {
            WireShape::IntPcm { .. } => FMT_PCM,
            WireShape::FloatPcm { .. } => FMT_IEEE_FLOAT,
            WireShape::Alaw => FMT_ALAW,
            WireShape::Mulaw => FMT_MULAW,
        }
    }

    fn well_known_guid(self) -> [u8; 16] {
        match self {
            WireShape::IntPcm { .. } => GUID_PCM,
            WireShape::FloatPcm { .. } => GUID_IEEE_FLOAT,
            WireShape::Alaw => GUID_ALAW,
            WireShape::Mulaw => GUID_MULAW,
        }
    }
}

fn wire_shape_for_params(p: &CodecParameters) -> Result<WireShape> {
    match p.codec_id.as_str() {
        "pcm_alaw" => return Ok(WireShape::Alaw),
        "pcm_mulaw" => return Ok(WireShape::Mulaw),
        _ => {}
    }
    let fmt = p
        .sample_format
        .or_else(|| super::pcm::sample_format_for(&p.codec_id))
        .ok_or_else(|| Error::unsupported(format!("WAV: unknown PCM codec {}", p.codec_id)))?;
    Ok(match fmt {
        SampleFormat::U8 => WireShape::IntPcm { bits: 8 },
        SampleFormat::S16 => WireShape::IntPcm { bits: 16 },
        SampleFormat::S24 => WireShape::IntPcm { bits: 24 },
        SampleFormat::S32 => WireShape::IntPcm { bits: 32 },
        SampleFormat::F32 => WireShape::FloatPcm { bits: 32 },
        SampleFormat::F64 => WireShape::FloatPcm { bits: 64 },
        other => {
            return Err(Error::unsupported(format!(
                "WAV muxer cannot write sample format {:?}",
                other
            )));
        }
    })
}

struct WavMuxer {
    output: Box<dyn WriteSeek>,
    channels: u16,
    sample_rate: u32,
    shape: WireShape,
    extensible: Option<ExtensibleOpts>,
    riff_size_offset: u64,
    data_size_offset: u64,
    data_bytes: u64,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for WavMuxer {
    fn format_name(&self) -> &str {
        "wav"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("WAV header already written"));
        }
        let bits_per_sample = self.shape.bits_per_sample();
        let block_align = (bits_per_sample / 8) * self.channels;
        let byte_rate = self.sample_rate * block_align as u32;

        // On-wire wFormatTag: caller's extensible opt-in overrides the
        // per-codec default (the underlying shape still drives
        // wBitsPerSample / block_align / byte_rate).
        let format_tag = if self.extensible.is_some() {
            FMT_EXTENSIBLE
        } else {
            self.shape.format_tag()
        };

        self.output.write_all(b"RIFF")?;
        self.riff_size_offset = self.output.stream_position()?;
        self.output.write_all(&0u32.to_le_bytes())?; // placeholder
        self.output.write_all(b"WAVE")?;

        // fmt chunk: 16 bytes for plain WAVEFORMAT, 40 bytes for
        // WAVEFORMATEXTENSIBLE (16 + 2 cbSize + 22 ext).
        let fmt_size: u32 = if self.extensible.is_some() { 40 } else { 16 };
        self.output.write_all(b"fmt ")?;
        self.output.write_all(&fmt_size.to_le_bytes())?;
        self.output.write_all(&format_tag.to_le_bytes())?;
        self.output.write_all(&self.channels.to_le_bytes())?;
        self.output.write_all(&self.sample_rate.to_le_bytes())?;
        self.output.write_all(&byte_rate.to_le_bytes())?;
        self.output.write_all(&block_align.to_le_bytes())?;
        self.output.write_all(&bits_per_sample.to_le_bytes())?;

        if let Some(opts) = &self.extensible {
            // cbSize (22) + 22-byte extension. Layout per
            // docs/container/riff/waveformatextensible/README.md
            // §"Structure layout".
            self.output.write_all(&22u16.to_le_bytes())?;
            let valid = opts.valid_bits_per_sample.unwrap_or(bits_per_sample);
            self.output.write_all(&valid.to_le_bytes())?;
            self.output.write_all(&opts.channel_mask.to_le_bytes())?;
            let guid = opts
                .subformat
                .unwrap_or_else(|| self.shape.well_known_guid());
            self.output.write_all(&guid)?;
        }

        self.output.write_all(b"data")?;
        self.data_size_offset = self.output.stream_position()?;
        self.output.write_all(&0u32.to_le_bytes())?; // placeholder

        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("WAV muxer: write_header not called"));
        }
        self.output.write_all(&packet.data)?;
        self.data_bytes += packet.data.len() as u64;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        // Pad data chunk to even length.
        if self.data_bytes % 2 == 1 {
            self.output.write_all(&[0u8])?;
        }
        let end = self.output.stream_position()?;

        // Patch "data" chunk size.
        let data_size_u32: u32 = self
            .data_bytes
            .try_into()
            .map_err(|_| Error::other("WAV data chunk exceeds 4 GiB"))?;
        self.output.seek(SeekFrom::Start(self.data_size_offset))?;
        self.output.write_all(&data_size_u32.to_le_bytes())?;

        // Patch "RIFF" size: total file size minus 8 (RIFF + size fields).
        let riff_size_u32: u32 = (end - 8)
            .try_into()
            .map_err(|_| Error::other("WAV RIFF size exceeds 4 GiB"))?;
        self.output.seek(SeekFrom::Start(self.riff_size_offset))?;
        self.output.write_all(&riff_size_u32.to_le_bytes())?;

        self.output.seek(SeekFrom::Start(end))?;
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecParameters, MediaType};

    fn make_stream(fmt: SampleFormat, ch: u16, sr: u32) -> StreamInfo {
        let mut params = CodecParameters::audio(super::super::pcm::codec_id_for(fmt).unwrap());
        params.media_type = MediaType::Audio;
        params.channels = Some(ch);
        params.sample_rate = Some(sr);
        params.sample_format = Some(fmt);
        StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, sr as i64),
            duration: None,
            start_time: Some(0),
            params,
        }
    }

    fn wave_format_tag(p: &CodecParameters) -> Option<u16> {
        match p.tag.as_ref()? {
            oxideav_core::CodecTag::WaveFormat(t) => Some(*t),
            _ => None,
        }
    }

    fn make_g711_stream(codec: &str, ch: u16, sr: u32) -> StreamInfo {
        let mut params = CodecParameters::audio(CodecId::new(codec));
        params.media_type = MediaType::Audio;
        params.channels = Some(ch);
        params.sample_rate = Some(sr);
        // G.711 expands to S16 once decoded — sample_format describes
        // the post-decode shape, matching the round-75 oxideav-avi
        // convention. The on-wire packets are 8-bit codewords.
        params.sample_format = Some(SampleFormat::S16);
        StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, sr as i64),
            duration: None,
            start_time: Some(0),
            params,
        }
    }

    /// Mux a single-packet stream through `open_muxer_with` to a
    /// uniquely-named tmpfile, then return both the encoded bytes and
    /// an open demuxer over them. The tmpfile is removed after the
    /// read. Avoids `Cursor<&mut Vec<u8>>` lifetime traps —
    /// `Box<dyn WriteSeek>` requires `'static`.
    fn mux_to_bytes(
        stream: &StreamInfo,
        payload: &[u8],
        opts: WavMuxOptions,
        tag: &str,
    ) -> Vec<u8> {
        let tmp = std::env::temp_dir().join(format!("oxideav-basic-wav-r77-{tag}.wav"));
        let _ = std::fs::remove_file(&tmp);
        {
            let f = std::fs::File::create(&tmp).unwrap();
            let ws: Box<dyn WriteSeek> = Box::new(f);
            let mut mux = open_muxer_with(ws, std::slice::from_ref(stream), opts).unwrap();
            mux.write_header().unwrap();
            let pkt = Packet::new(0, stream.time_base, payload.to_vec());
            mux.write_packet(&pkt).unwrap();
            mux.write_trailer().unwrap();
        }
        let bytes = std::fs::read(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        bytes
    }

    fn open_demux_from_bytes(bytes: Vec<u8>) -> Box<dyn Demuxer> {
        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
        open_demuxer(rs, &oxideav_core::NullCodecResolver).unwrap()
    }

    #[test]
    fn round_trip_s16_mono() {
        // Write then read back a small S16 mono WAV via the public demuxer/muxer paths.
        let samples: Vec<i16> = (0..1000).map(|i| ((i * 32) - 16000) as i16).collect();
        let mut payload = Vec::with_capacity(samples.len() * 2);
        for s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }

        // Mux to a temp file, then demux and compare.
        let stream = make_stream(SampleFormat::S16, 1, 48_000);
        let tmp = std::env::temp_dir().join("oxideav-basic-wav-test.wav");
        {
            let f = std::fs::File::create(&tmp).unwrap();
            let ws: Box<dyn WriteSeek> = Box::new(f);
            let mut mux = open_muxer(ws, std::slice::from_ref(&stream)).unwrap();
            mux.write_header().unwrap();
            let pkt = Packet::new(0, stream.time_base, payload.clone());
            mux.write_packet(&pkt).unwrap();
            mux.write_trailer().unwrap();
        }
        let rs: Box<dyn ReadSeek> = Box::new(std::fs::File::open(&tmp).unwrap());
        let mut dmx = open_demuxer(rs, &oxideav_core::NullCodecResolver).unwrap();
        assert_eq!(dmx.format_name(), "wav");
        assert_eq!(dmx.streams().len(), 1);
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
        let mut out_bytes = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out_bytes.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out_bytes, payload);
    }

    /// `WAVE_FORMAT_ALAW (0x0006)` — codec_id is `pcm_alaw`, on-wire
    /// stays 8-bit codewords, sample_format hint is S16 (decoded shape).
    #[test]
    fn round_trip_alaw_mono() {
        // Synthetic A-law codewords — A-law is a byte stream, so any
        // distinct-byte sequence exercises the byte-for-byte plumbing.
        let payload: Vec<u8> = (0..=255u8).collect();
        let stream = make_g711_stream("pcm_alaw", 1, 8_000);
        let bytes = mux_to_bytes(&stream, &payload, WavMuxOptions::default(), "alaw-mono");
        let mut dmx = open_demux_from_bytes(bytes);
        assert_eq!(dmx.streams().len(), 1);
        let s = &dmx.streams()[0];
        assert_eq!(s.params.codec_id, CodecId::new("pcm_alaw"));
        assert_eq!(wave_format_tag(&s.params), Some(FMT_ALAW));
        assert_eq!(s.params.sample_format, Some(SampleFormat::S16));
        // bit_rate is the on-wire rate (8 bits/sample * channels * rate),
        // NOT the post-decode S16 rate.
        assert_eq!(s.params.bit_rate, Some(8 * 8_000));
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, payload);
    }

    /// `WAVE_FORMAT_MULAW (0x0007)` — codec_id is `pcm_mulaw`.
    #[test]
    fn round_trip_mulaw_stereo() {
        let payload: Vec<u8> = (0..512u32).map(|i| (i & 0xFF) as u8).collect();
        let stream = make_g711_stream("pcm_mulaw", 2, 8_000);
        let bytes = mux_to_bytes(&stream, &payload, WavMuxOptions::default(), "mulaw-stereo");
        let mut dmx = open_demux_from_bytes(bytes);
        let s = &dmx.streams()[0];
        assert_eq!(s.params.codec_id, CodecId::new("pcm_mulaw"));
        assert_eq!(wave_format_tag(&s.params), Some(FMT_MULAW));
        // stereo G.711 — block_align = 2 bytes (1 byte/channel),
        // byte_rate = 16000, bit_rate = 128 kbps.
        assert_eq!(s.params.bit_rate, Some(16 * 8_000));
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, payload);
    }

    /// `WAVE_FORMAT_EXTENSIBLE (0xFFFE)` end-to-end — muxer emits the
    /// 40-byte fmt chunk with cbSize = 22, demuxer parses the
    /// extension and exposes channel_mask / valid_bits / SubFormat via
    /// both metadata keys AND the typed accessors.
    #[test]
    fn round_trip_extensible_5_1_pcm() {
        // 6-channel 5.1-Microsoft layout (FL FR FC LFE BL BR) per
        // docs/container/riff/waveformatextensible/README.md
        // §"Channel-mask channel ordering". `payload` is one frame of
        // 6 distinct s16 samples so we can verify frame boundary
        // alignment after the round trip.
        const MASK_5_1: u32 = 0x0003F;
        let frame: [i16; 6] = [-100, 200, -300, 400, -500, 600];
        let mut payload = Vec::new();
        for _ in 0..32 {
            for s in &frame {
                payload.extend_from_slice(&s.to_le_bytes());
            }
        }

        let mut stream = make_stream(SampleFormat::S16, 6, 48_000);
        // Build via open_muxer_with so the muxer emits the EXTENSIBLE
        // fmt chunk. open_muxer (the registry default) writes the
        // legacy 16-byte fmt chunk for maximum compatibility.
        stream.params.codec_id = CodecId::new("pcm_s16le");
        let opts = WavMuxOptions::default().with_extensible(MASK_5_1);
        let buf = mux_to_bytes(&stream, &payload, opts, "ext-5-1-pcm");

        // Sanity-check the on-wire fmt chunk size + cbSize.
        // RIFF(4)+size(4)+WAVE(4)+"fmt "(4)+size(4) == 20 bytes
        // before the fmt body; fmt size at offset 16 must be 40.
        assert_eq!(&buf[12..16], b"fmt ");
        let fmt_size = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
        assert_eq!(fmt_size, 40, "EXTENSIBLE fmt chunk must be 40 bytes");
        // wFormatTag at fmt body[0..2] == 0xFFFE
        assert_eq!(u16::from_le_bytes([buf[20], buf[21]]), FMT_EXTENSIBLE);
        // cbSize at fmt body[16..18] == 22
        assert_eq!(u16::from_le_bytes([buf[36], buf[37]]), 22);

        // Demux it back.
        let dmx = open_demux_from_bytes(buf);
        // SubFormat == KSDATAFORMAT_SUBTYPE_PCM resolves the codec id
        // to the legacy pcm_s16le (the round-75 oxideav-avi convention
        // — the GUID identifies the codec, not the bit depth).
        let s = &dmx.streams()[0];
        assert_eq!(s.params.codec_id, CodecId::new("pcm_s16le"));
        assert_eq!(wave_format_tag(&s.params), Some(FMT_EXTENSIBLE));

        // Metadata round-trip.
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:fmt.channel_mask"),
            Some(&format!("0x{MASK_5_1:08X}"))
        );
        assert_eq!(
            md.get("wav:fmt.valid_bits_per_sample"),
            Some(&"16".to_string())
        );
        assert_eq!(
            md.get("wav:fmt.subformat"),
            Some(&"00000001-0000-0010-8000-00AA00389B71".to_string())
        );
    }

    /// Typed accessors on the concrete `WavDemuxer` carry the same
    /// EXTENSIBLE side-info the metadata keys do.
    #[test]
    fn extensible_accessors_match_metadata() {
        const MASK_STEREO: u32 = 0x00003;
        let payload = vec![0u8; 4 * 100]; // 100 stereo s16 frames

        let mut stream = make_stream(SampleFormat::S16, 2, 44_100);
        stream.params.codec_id = CodecId::new("pcm_s16le");

        let opts = WavMuxOptions::default().with_extensible(MASK_STEREO);
        let bytes = mux_to_bytes(&stream, &payload, opts, "ext-stereo-md");
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:fmt.channel_mask"),
            Some(&format!("0x{MASK_STEREO:08X}"))
        );
        assert_eq!(
            md.get("wav:fmt.valid_bits_per_sample"),
            Some(&"16".to_string())
        );
    }

    /// EXTENSIBLE muxing of A-law: muxer writes `wFormatTag = 0xFFFE`
    /// with the KSDATAFORMAT_SUBTYPE_ALAW GUID; demuxer resolves
    /// codec_id to `pcm_alaw` via the GUID path.
    #[test]
    fn extensible_alaw_through_guid() {
        let payload: Vec<u8> = (0..=255u8).collect();
        let stream = make_g711_stream("pcm_alaw", 1, 8_000);

        let opts = WavMuxOptions::default().with_extensible(0x00004); // FRONT_CENTER
        let bytes = mux_to_bytes(&stream, &payload, opts, "ext-alaw");
        let mut dmx = open_demux_from_bytes(bytes);
        let s = &dmx.streams()[0];
        // The legacy wFormatTag is preserved (0xFFFE) but the codec_id
        // resolves through the SubFormat GUID — the demuxer must
        // dispatch G.711 even though wFormatTag is the EXTENSIBLE
        // escape hatch.
        assert_eq!(s.params.codec_id, CodecId::new("pcm_alaw"));
        assert_eq!(wave_format_tag(&s.params), Some(FMT_EXTENSIBLE));
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, payload);
    }

    /// EXTENSIBLE with an unknown SubFormat GUID — the demuxer must
    /// synthesise a `wav:guid_<canonical>` codec id so downstream
    /// `make_decoder` lookups fail naming the actual GUID rather than
    /// the opaque `0xFFFE` tag.
    #[test]
    fn extensible_unknown_guid_synthesised_id() {
        // Hand-build a minimal EXTENSIBLE WAV with a bogus GUID, then
        // feed it to the demuxer. We bypass the muxer so we can pick
        // an unknown SubFormat directly.
        let bogus_guid: [u8; 16] = [
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC,
            0xDE, 0xF0,
        ];
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes()); // riff size placeholder
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&40u32.to_le_bytes()); // fmt size
        buf.extend_from_slice(&FMT_EXTENSIBLE.to_le_bytes()); // wFormatTag
        buf.extend_from_slice(&1u16.to_le_bytes()); // channels
        buf.extend_from_slice(&44_100u32.to_le_bytes()); // sample_rate
        buf.extend_from_slice(&88_200u32.to_le_bytes()); // byte_rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block_align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits_per_sample
        buf.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        buf.extend_from_slice(&16u16.to_le_bytes()); // wValidBitsPerSample
        buf.extend_from_slice(&0x00004u32.to_le_bytes()); // dwChannelMask (FC)
        buf.extend_from_slice(&bogus_guid); // SubFormat
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes()); // empty data chunk

        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(buf));
        let dmx_res = open_demuxer(rs, &oxideav_core::NullCodecResolver);
        let dmx = dmx_res.expect("unknown-GUID extensible stream still parses");
        let id = dmx.streams()[0].params.codec_id.as_str().to_string();
        assert!(
            id.starts_with("wav:guid_"),
            "unknown GUID must synthesise wav:guid_<text>, got {id:?}"
        );
        // Canonical text form: lowercase hex-dump in the on-wire
        // little-endian first-three-groups layout.
        assert!(
            id.contains("EFBEADDE-FECA-BEBA"),
            "synthesised id must carry the canonical GUID text, got {id:?}"
        );
    }

    /// `WAVE_FORMAT_EXTENSIBLE` with cbSize < 22 must reject — the spec
    /// mandates a 22-byte extension. Per
    /// docs/container/riff/waveformatextensible/README.md §"Structure
    /// layout": "RIFF `fmt` chunks carrying it are 40 bytes (38 bytes
    /// of struct payload + 2-byte `cbSize = 22`)."
    #[test]
    fn extensible_short_cbsize_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        // 40-byte fmt body but cbSize = 10 (insufficient).
        buf.extend_from_slice(&40u32.to_le_bytes());
        buf.extend_from_slice(&FMT_EXTENSIBLE.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&44_100u32.to_le_bytes());
        buf.extend_from_slice(&88_200u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(&10u16.to_le_bytes()); // cbSize too small
        buf.extend_from_slice(&[0u8; 20]); // padding to reach 40 fmt body bytes
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());

        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(buf));
        let err = open_demuxer(rs, &oxideav_core::NullCodecResolver).err();
        assert!(
            matches!(err, Some(Error::InvalidData(_))),
            "short cbSize must yield Error::InvalidData, got {err:?}"
        );
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `bext`
    /// chunk body inserted between `fmt ` and `data`. Returns the file
    /// bytes ready for `open_demux_from_bytes`. RIFF size is left at 0
    /// (the demuxer doesn't validate it).
    fn wav_with_bext(bext_body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // channels
        buf.extend_from_slice(&8_000u32.to_le_bytes()); // sample_rate
        buf.extend_from_slice(&16_000u32.to_le_bytes()); // byte_rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block_align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits_per_sample
                                                     // bext chunk.
        buf.extend_from_slice(b"bext");
        buf.extend_from_slice(&(bext_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(bext_body);
        if bext_body.len() % 2 == 1 {
            buf.push(0); // RIFF word-alignment pad
        }
        // empty data chunk.
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Assemble a 602-byte BWF v2 `bext` fixed struct plus a coding
    /// history string. Field offsets follow EBU Tech 3285 v2 §2.3.
    #[allow(clippy::too_many_arguments)]
    fn make_bext_v2(
        description: &str,
        originator: &str,
        originator_ref: &str,
        date: &str,
        time: &str,
        time_reference: u64,
        umid: &[u8; 64],
        loudness: [i16; 5],
        coding_history: &str,
    ) -> Vec<u8> {
        fn put_ascii(buf: &mut Vec<u8>, s: &str, width: usize) {
            let bytes = s.as_bytes();
            let n = bytes.len().min(width);
            buf.extend_from_slice(&bytes[..n]);
            buf.resize(buf.len() + (width - n), 0);
        }
        let mut b = Vec::new();
        put_ascii(&mut b, description, 256);
        put_ascii(&mut b, originator, 32);
        put_ascii(&mut b, originator_ref, 32);
        put_ascii(&mut b, date, 10);
        put_ascii(&mut b, time, 8);
        b.extend_from_slice(&(time_reference as u32).to_le_bytes()); // low
        b.extend_from_slice(&((time_reference >> 32) as u32).to_le_bytes()); // high
        b.extend_from_slice(&2u16.to_le_bytes()); // Version = 2
        b.extend_from_slice(umid);
        for v in &loudness {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.resize(BEXT_FIXED_LEN, 0); // Reserved[180] zero-fill to 602
        assert_eq!(b.len(), BEXT_FIXED_LEN);
        b.extend_from_slice(coding_history.as_bytes());
        b
    }

    /// `fmt_loudness` mirrors the EBU Tech 3285 v2 §2.4 round-to-nearest
    /// fixed-point: the stored WORD is `round(100 × value)`, so dividing
    /// by 100 with two decimals recovers the displayed value. The §2.4
    /// negative-number examples (`-22.64` / `-22.65` / `-22.66` stored
    /// as `-2264` / `-2265` / `-2265`) are exercised in reverse here.
    #[test]
    fn bext_loudness_formatting() {
        assert_eq!(fmt_loudness(-2264), "-22.64");
        assert_eq!(fmt_loudness(-2265), "-22.65");
        assert_eq!(fmt_loudness(0), "0.00");
        assert_eq!(fmt_loudness(2300), "23.00");
        assert_eq!(fmt_loudness(-50), "-0.50"); // sub-unity negative keeps the sign
        assert_eq!(fmt_loudness(7), "0.07");
        assert_eq!(fmt_loudness(-1), "-0.01");
    }

    /// Full BWF v2 `bext` round-trip: every field surfaces under its
    /// `wav:bext.*` metadata key, loudness fields decode to two
    /// decimals, the 64-bit TimeReference reassembles, and the UMID is
    /// hex-encoded.
    #[test]
    fn bext_v2_full_metadata() {
        let mut umid = [0u8; 64];
        umid[0] = 0x06;
        umid[1] = 0x0a;
        umid[63] = 0xff;
        let body = make_bext_v2(
            "Scene 1 take 3",
            "OxideAV Recorder",
            "USABC2400001",
            "2026-05-23",
            "14:30:00",
            // 48000 samples/s × 90061 s ≈ a value that spans 32 bits.
            0x0000_0001_2345_6789,
            &umid,
            [-2305, 700, -120, -1850, -2010],
            "A=PCM,F=48000,W=24,M=stereo,T=OxideAV\r\n",
        );
        let bytes = wav_with_bext(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(
            md.get("wav:bext.description"),
            Some(&"Scene 1 take 3".to_string())
        );
        assert_eq!(
            md.get("wav:bext.originator"),
            Some(&"OxideAV Recorder".to_string())
        );
        assert_eq!(
            md.get("wav:bext.originator_reference"),
            Some(&"USABC2400001".to_string())
        );
        assert_eq!(
            md.get("wav:bext.origination_date"),
            Some(&"2026-05-23".to_string())
        );
        assert_eq!(
            md.get("wav:bext.origination_time"),
            Some(&"14:30:00".to_string())
        );
        assert_eq!(
            md.get("wav:bext.time_reference"),
            Some(&0x0000_0001_2345_6789u64.to_string())
        );
        assert_eq!(md.get("wav:bext.version"), Some(&"2".to_string()));
        assert_eq!(
            md.get("wav:bext.loudness_value"),
            Some(&"-23.05".to_string())
        );
        assert_eq!(md.get("wav:bext.loudness_range"), Some(&"7.00".to_string()));
        assert_eq!(
            md.get("wav:bext.max_true_peak_level"),
            Some(&"-1.20".to_string())
        );
        assert_eq!(
            md.get("wav:bext.max_momentary_loudness"),
            Some(&"-18.50".to_string())
        );
        assert_eq!(
            md.get("wav:bext.max_short_term_loudness"),
            Some(&"-20.10".to_string())
        );
        // UMID hex begins with the bytes we set and ends with 0xff.
        let umid_hex = md.get("wav:bext.umid").expect("umid present");
        assert!(umid_hex.starts_with("060a"), "umid hex {umid_hex:?}");
        assert!(umid_hex.ends_with("ff"), "umid hex {umid_hex:?}");
        assert_eq!(umid_hex.len(), 128); // 64 bytes × 2 hex chars
        assert_eq!(
            md.get("wav:bext.coding_history"),
            Some(&"A=PCM,F=48000,W=24,M=stereo,T=OxideAV".to_string())
        );
    }

    /// BWF v0 `bext`: no UMID, no loudness fields. Version = 0 so the
    /// loudness keys must be absent and the (all-zero) UMID suppressed,
    /// while the text + TimeReference fields still surface.
    #[test]
    fn bext_v0_omits_umid_and_loudness() {
        // 602-byte fixed struct, version 0, all UMID/loudness zero.
        let mut body = vec![0u8; BEXT_FIXED_LEN];
        // Description "Field recording".
        let desc = b"Field recording";
        body[..desc.len()].copy_from_slice(desc);
        // TimeReferenceLow = 12345 (offset 338).
        body[338..342].copy_from_slice(&12_345u32.to_le_bytes());
        // Version = 0 (offset 346) — already zero.
        let bytes = wav_with_bext(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(
            md.get("wav:bext.description"),
            Some(&"Field recording".to_string())
        );
        assert_eq!(md.get("wav:bext.version"), Some(&"0".to_string()));
        assert_eq!(
            md.get("wav:bext.time_reference"),
            Some(&"12345".to_string())
        );
        // v0 must not emit loudness or UMID.
        assert!(!md.contains_key("wav:bext.umid"));
        assert!(!md.contains_key("wav:bext.loudness_value"));
        assert!(!md.contains_key("wav:bext.max_true_peak_level"));
    }

    /// A `bext` chunk shorter than the 602-byte fixed struct is
    /// malformed; the parser must skip it without panicking and the
    /// stream must still open (the chunk is treated as opaque).
    #[test]
    fn bext_truncated_is_skipped() {
        let body = vec![0u8; 100]; // < 602
        let bytes = wav_with_bext(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(md.keys().all(|k| !k.starts_with("wav:bext.")));
        // Stream still resolves to PCM s16.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `cue `
    /// chunk and optional `LIST adtl` body inserted between `fmt ` and
    /// `data`. Returns the file bytes ready for `open_demux_from_bytes`.
    fn wav_with_cue_and_adtl(cue_body: &[u8], adtl_body: Option<&[u8]>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // cue chunk
        buf.extend_from_slice(b"cue ");
        buf.extend_from_slice(&(cue_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(cue_body);
        if cue_body.len() % 2 == 1 {
            buf.push(0);
        }
        // optional LIST adtl chunk
        if let Some(adtl) = adtl_body {
            // LIST chunk wraps a 4-byte form-type 'adtl' + sub-chunks.
            buf.extend_from_slice(b"LIST");
            buf.extend_from_slice(&((adtl.len() + 4) as u32).to_le_bytes());
            buf.extend_from_slice(b"adtl");
            buf.extend_from_slice(adtl);
            if (adtl.len() + 4) % 2 == 1 {
                buf.push(0);
            }
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Build a single 24-byte `<cue-point>` record per
    /// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3.
    fn cue_point(
        dw_name: u32,
        dw_position: u32,
        fcc_chunk: &[u8; 4],
        dw_chunk_start: u32,
        dw_block_start: u32,
        dw_sample_offset: u32,
    ) -> Vec<u8> {
        let mut b = Vec::with_capacity(24);
        b.extend_from_slice(&dw_name.to_le_bytes());
        b.extend_from_slice(&dw_position.to_le_bytes());
        b.extend_from_slice(fcc_chunk);
        b.extend_from_slice(&dw_chunk_start.to_le_bytes());
        b.extend_from_slice(&dw_block_start.to_le_bytes());
        b.extend_from_slice(&dw_sample_offset.to_le_bytes());
        b
    }

    /// Build a `labl` or `note` sub-chunk body (without the chunk
    /// header) per RIFF MCI §3 "Label and Note Information".
    fn adtl_text_subchunk(id: &[u8; 4], dw_name: u32, text: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(id);
        // Body = 4-byte dwName + ZSTR (text + NUL terminator).
        let body_len = 4 + text.len() + 1;
        b.extend_from_slice(&(body_len as u32).to_le_bytes());
        b.extend_from_slice(&dw_name.to_le_bytes());
        b.extend_from_slice(text.as_bytes());
        b.push(0);
        if body_len % 2 == 1 {
            b.push(0);
        }
        b
    }

    /// Full `cue ` + `LIST adtl` round-trip: two cue points, each with
    /// a `labl` and a `note`, surface under the documented metadata
    /// keys.
    #[test]
    fn cue_and_adtl_full_metadata() {
        // Two cue points (id=1 at sample 0, id=2 at sample 12345).
        let mut cue_body = Vec::new();
        cue_body.extend_from_slice(&2u32.to_le_bytes()); // dwCuePoints
        cue_body.extend(cue_point(1, 0, b"data", 0, 0, 0));
        cue_body.extend(cue_point(2, 12_345, b"data", 0, 0, 12_345));

        // LIST adtl with labl + note for each cue point.
        let mut adtl = Vec::new();
        adtl.extend(adtl_text_subchunk(b"labl", 1, "Intro"));
        adtl.extend(adtl_text_subchunk(b"note", 1, "Fade-in"));
        adtl.extend(adtl_text_subchunk(b"labl", 2, "Verse"));
        adtl.extend(adtl_text_subchunk(b"note", 2, "Vocal entry"));

        let bytes = wav_with_cue_and_adtl(&cue_body, Some(&adtl));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(md.get("wav:cue.count"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:cue.1.position"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:cue.1.fcc_chunk"), Some(&"data".to_string()));
        assert_eq!(md.get("wav:cue.1.chunk_start"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:cue.1.block_start"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:cue.1.sample_offset"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:cue.2.position"), Some(&"12345".to_string()));
        assert_eq!(
            md.get("wav:cue.2.sample_offset"),
            Some(&"12345".to_string())
        );

        assert_eq!(md.get("wav:adtl.labl.1"), Some(&"Intro".to_string()));
        assert_eq!(md.get("wav:adtl.note.1"), Some(&"Fade-in".to_string()));
        assert_eq!(md.get("wav:adtl.labl.2"), Some(&"Verse".to_string()));
        assert_eq!(md.get("wav:adtl.note.2"), Some(&"Vocal entry".to_string()));
    }

    /// `ltxt` sub-chunk surfaces dwSampleLength, FOURCC purpose, and
    /// text under `wav:adtl.ltxt.<dwName>.*`.
    #[test]
    fn adtl_ltxt_segment_metadata() {
        // Single cue point so the ltxt reference is meaningful.
        let mut cue_body = Vec::new();
        cue_body.extend_from_slice(&1u32.to_le_bytes());
        cue_body.extend(cue_point(7, 1000, b"data", 0, 0, 1000));

        // ltxt chunk for cue 7: 4410-sample segment, 'scrp' (script) text.
        let mut ltxt_body = Vec::new();
        ltxt_body.extend_from_slice(&7u32.to_le_bytes()); // dwName
        ltxt_body.extend_from_slice(&4410u32.to_le_bytes()); // dwSampleLength
        ltxt_body.extend_from_slice(b"scrp"); // dwPurpose
        ltxt_body.extend_from_slice(&0u16.to_le_bytes()); // wCountry
        ltxt_body.extend_from_slice(&0u16.to_le_bytes()); // wLanguage
        ltxt_body.extend_from_slice(&0u16.to_le_bytes()); // wDialect
        ltxt_body.extend_from_slice(&0u16.to_le_bytes()); // wCodePage
        ltxt_body.extend_from_slice(b"Hello world");
        ltxt_body.push(0);

        let mut adtl = Vec::new();
        adtl.extend_from_slice(b"ltxt");
        adtl.extend_from_slice(&(ltxt_body.len() as u32).to_le_bytes());
        adtl.extend_from_slice(&ltxt_body);
        if ltxt_body.len() % 2 == 1 {
            adtl.push(0);
        }

        let bytes = wav_with_cue_and_adtl(&cue_body, Some(&adtl));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(md.get("wav:adtl.ltxt.7.length"), Some(&"4410".to_string()));
        assert_eq!(md.get("wav:adtl.ltxt.7.purpose"), Some(&"scrp".to_string()));
        assert_eq!(
            md.get("wav:adtl.ltxt.7.text"),
            Some(&"Hello world".to_string())
        );
    }

    /// A `cue ` chunk whose `dwCuePoints` count exceeds the body length
    /// must not panic — the parser surfaces only the records that
    /// actually fit in the body.
    #[test]
    fn cue_truncated_count_is_clamped() {
        // Claim 5 points, ship 1.
        let mut cue_body = Vec::new();
        cue_body.extend_from_slice(&5u32.to_le_bytes());
        cue_body.extend(cue_point(42, 100, b"data", 0, 0, 100));
        let bytes = wav_with_cue_and_adtl(&cue_body, None);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cue.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:cue.42.position"), Some(&"100".to_string()));
        // Stream still opens cleanly.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// An `adtl` list without a matching `cue ` chunk still emits its
    /// `wav:adtl.*` keys — the spec doesn't require the cue chunk to
    /// precede the adtl list, and downstream consumers can cross-
    /// reference dwName values themselves.
    #[test]
    fn adtl_without_cue_still_surfaces() {
        let mut adtl = Vec::new();
        adtl.extend(adtl_text_subchunk(b"labl", 99, "Orphan label"));
        // cue body with zero points exercises the count=0 path.
        let cue_body = 0u32.to_le_bytes().to_vec();
        let bytes = wav_with_cue_and_adtl(&cue_body, Some(&adtl));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cue.count"), Some(&"0".to_string()));
        assert_eq!(
            md.get("wav:adtl.labl.99"),
            Some(&"Orphan label".to_string())
        );
    }

    /// Build a minimal valid PCM WAV with caller-supplied raw `smpl`
    /// and/or `inst` chunks inserted between `fmt ` and `data`. Mirrors
    /// `wav_with_cue_and_adtl` but for the sampler/instrument chunk
    /// pair.
    fn wav_with_smpl_and_inst(smpl_body: Option<&[u8]>, inst_body: Option<&[u8]>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        if let Some(smpl) = smpl_body {
            buf.extend_from_slice(b"smpl");
            buf.extend_from_slice(&(smpl.len() as u32).to_le_bytes());
            buf.extend_from_slice(smpl);
            if smpl.len() % 2 == 1 {
                buf.push(0);
            }
        }
        if let Some(inst) = inst_body {
            buf.extend_from_slice(b"inst");
            buf.extend_from_slice(&(inst.len() as u32).to_le_bytes());
            buf.extend_from_slice(inst);
            if inst.len() % 2 == 1 {
                buf.push(0);
            }
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Build a `smpl` fixed header (36 bytes) followed by N sample-loop
    /// records (24 bytes each).
    #[allow(clippy::too_many_arguments)]
    fn smpl_body(
        manufacturer: u32,
        product: u32,
        sample_period: u32,
        midi_unity_note: u32,
        midi_pitch_fraction: u32,
        smpte_format: u32,
        smpte_offset: u32,
        c_sample_loops_claimed: u32,
        cb_sampler_data: u32,
        loops: &[(u32, u32, u32, u32, u32, u32)],
    ) -> Vec<u8> {
        let mut b = Vec::new();
        for v in [
            manufacturer,
            product,
            sample_period,
            midi_unity_note,
            midi_pitch_fraction,
            smpte_format,
            smpte_offset,
            c_sample_loops_claimed,
            cb_sampler_data,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for &(id, ty, start, end, frac, count) in loops {
            for v in [id, ty, start, end, frac, count] {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b
    }

    /// Full `smpl` round-trip: one loop, every fixed-header field and
    /// the per-loop record surface under the documented metadata keys.
    /// SMPTE offset is decoded as `HH:MM:SS:FF`.
    #[test]
    fn smpl_full_metadata() {
        // SMPTE offset 0x01020304 → 01:02:03:04 (HH MM SS FF).
        let body = smpl_body(
            0x1234,                   // manufacturer
            0xDEAD_BEEF,              // product
            22_675,                   // sample period (ns; ≈ 44.1 kHz)
            60,                       // MIDI middle-C
            0x8000_0000,              // MIDI pitch fraction (½ semitone)
            30,                       // SMPTE 30 fps
            0x01_02_03_04,            // SMPTE offset HH:MM:SS:FF
            1,                        // one sample loop
            0,                        // no sampler-specific data
            &[(7, 0, 0, 1000, 0, 0)], // cue id 7, fwd loop, 0..1000, infinite
        );
        let bytes = wav_with_smpl_and_inst(Some(&body), None);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:smpl.manufacturer"), Some(&"4660".to_string()));
        assert_eq!(md.get("wav:smpl.product"), Some(&"3735928559".to_string()));
        assert_eq!(md.get("wav:smpl.sample_period"), Some(&"22675".to_string()));
        assert_eq!(md.get("wav:smpl.midi_unity_note"), Some(&"60".to_string()));
        assert_eq!(
            md.get("wav:smpl.midi_pitch_fraction"),
            Some(&"2147483648".to_string())
        );
        assert_eq!(md.get("wav:smpl.smpte_format"), Some(&"30".to_string()));
        assert_eq!(
            md.get("wav:smpl.smpte_offset"),
            Some(&"01:02:03:04".to_string())
        );
        assert_eq!(md.get("wav:smpl.sampler_data_len"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:smpl.num_sample_loops"), Some(&"1".to_string()));
        assert_eq!(
            md.get("wav:smpl.loop.0.cue_point_id"),
            Some(&"7".to_string())
        );
        assert_eq!(md.get("wav:smpl.loop.0.type"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:smpl.loop.0.start"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:smpl.loop.0.end"), Some(&"1000".to_string()));
        assert_eq!(md.get("wav:smpl.loop.0.fraction"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:smpl.loop.0.play_count"), Some(&"0".to_string()));
        // Stream still opens cleanly.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `smpl` chunk whose `cSampleLoops` count exceeds the records
    /// that actually fit in the chunk body is clamped to the records
    /// the body carries — defensive vs. writers that lie about the
    /// count (mirrors the `cue ` chunk's clamping behaviour).
    #[test]
    fn smpl_loop_count_clamped_to_body() {
        // Claim 5 loops but provide only 1 — parser must surface
        // num_sample_loops=1.
        let body = smpl_body(
            0,
            0,
            0,
            60,
            0,
            0,
            0,
            /* claim */ 5,
            0,
            &[(1, 0, 0, 100, 0, 0)],
        );
        let bytes = wav_with_smpl_and_inst(Some(&body), None);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:smpl.num_sample_loops"), Some(&"1".to_string()));
        // Only loop.0 should be present; loop.1.. must not be emitted.
        assert!(md.contains_key("wav:smpl.loop.0.cue_point_id"));
        assert!(!md.contains_key("wav:smpl.loop.1.cue_point_id"));
        assert!(!md.contains_key("wav:smpl.loop.4.cue_point_id"));
    }

    /// A `smpl` chunk shorter than the 36-byte fixed struct is
    /// malformed; the parser must skip it without panicking and the
    /// stream must still open.
    #[test]
    fn smpl_truncated_is_skipped() {
        let body = vec![0u8; 20]; // < 36
        let bytes = wav_with_smpl_and_inst(Some(&body), None);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(md.keys().all(|k| !k.starts_with("wav:smpl.")));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Full `inst` round-trip: signed `FineTune` / `Gain` are decoded
    /// as i8 (so `-1` shows as `-1`, not `255`), MIDI note fields are
    /// unsigned.
    #[test]
    fn inst_full_metadata() {
        // FineTune = -3 cents (0xFD), Gain = -6 dB (0xFA).
        let body: Vec<u8> = vec![60, 0xFD, 0xFA, 36, 96, 1, 127];
        let bytes = wav_with_smpl_and_inst(None, Some(&body));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:inst.unshifted_note"), Some(&"60".to_string()));
        assert_eq!(md.get("wav:inst.fine_tune"), Some(&"-3".to_string()));
        assert_eq!(md.get("wav:inst.gain"), Some(&"-6".to_string()));
        assert_eq!(md.get("wav:inst.low_note"), Some(&"36".to_string()));
        assert_eq!(md.get("wav:inst.high_note"), Some(&"96".to_string()));
        assert_eq!(md.get("wav:inst.low_velocity"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:inst.high_velocity"), Some(&"127".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// An `inst` chunk shorter than the 7-byte fixed struct is skipped
    /// as opaque and the stream still resolves.
    #[test]
    fn inst_truncated_is_skipped() {
        let body = vec![0u8; 5]; // < 7
        let bytes = wav_with_smpl_and_inst(None, Some(&body));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(md.keys().all(|k| !k.starts_with("wav:inst.")));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Both `smpl` and `inst` chunks present in the same file surface
    /// under their respective key namespaces without colliding. The
    /// odd-length `inst` chunk forces the 1-byte word-pad path; the
    /// `data` chunk that follows must still be located.
    #[test]
    fn smpl_and_inst_coexist_with_padding() {
        let smpl = smpl_body(0, 0, 0, 64, 0, 0, 0, 0, 0, &[]);
        let inst: Vec<u8> = vec![64, 0, 0, 0, 127, 1, 127]; // 7 bytes → odd
        let bytes = wav_with_smpl_and_inst(Some(&smpl), Some(&inst));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:smpl.midi_unity_note"), Some(&"64".to_string()));
        assert_eq!(md.get("wav:inst.unshifted_note"), Some(&"64".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// `fmt_guid` produces the canonical text form: first three groups
    /// little-endian, trailing two big-endian. This is the format
    /// `mmreg.h` definitions use (e.g. `KSDATAFORMAT_SUBTYPE_PCM` =
    /// `00000001-0000-0010-8000-00AA00389B71`).
    #[test]
    fn guid_canonical_text() {
        assert_eq!(fmt_guid(&GUID_PCM), "00000001-0000-0010-8000-00AA00389B71");
        assert_eq!(
            fmt_guid(&GUID_IEEE_FLOAT),
            "00000003-0000-0010-8000-00AA00389B71"
        );
        assert_eq!(fmt_guid(&GUID_ALAW), "00000006-0000-0010-8000-00AA00389B71");
        assert_eq!(
            fmt_guid(&GUID_MULAW),
            "00000007-0000-0010-8000-00AA00389B71"
        );
    }
}
