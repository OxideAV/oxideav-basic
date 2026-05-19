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

/// Parse a RIFF LIST chunk body. If the list type is `INFO`, map its
/// `IART`/`INAM`/... subchunks to standard key names (`artist`, `title`,
/// …) and append to `out`.
fn parse_list_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    if buf.len() < 4 {
        return;
    }
    if &buf[0..4] != b"INFO" {
        return;
    }
    let mut i = 4usize;
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
