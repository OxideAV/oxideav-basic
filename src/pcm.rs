//! PCM codec: uncompressed interleaved linear PCM.
//!
//! A PCM "codec" is trivial — the packet payload *is* the sample data. We
//! still funnel it through [`Decoder`] / [`Encoder`] so that pipelines treat it
//! uniformly.
//!
//! Codec IDs:
//! - `pcm_u8`   — unsigned 8-bit
//! - `pcm_s16le` — signed 16-bit little-endian
//! - `pcm_s24le` — signed 24-bit little-endian, packed
//! - `pcm_s32le` — signed 32-bit little-endian
//! - `pcm_f32le` — 32-bit IEEE float little-endian
//! - `pcm_f64le` — 64-bit IEEE float little-endian
//!
//! Asterisk-style signed-linear aliases:
//! - `slin`, `slin8`, `slin16`, `slin24`, `slin32`, `slin44`, `slin48`,
//!   `slin96`, `slin192` — all map onto the `pcm_s16le` implementation.
//!   The trailing digits only indicate the implied sample rate of the
//!   surrounding headerless `.sln*` container (see `slin.rs`); as a codec
//!   they are indistinguishable from `pcm_s16le`.

use oxideav_codec::{CodecInfo, CodecRegistry, Decoder, Encoder};
use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecParameters, CodecTag, Error, Frame, MediaType,
    Packet, ProbeContext, Result, SampleFormat, TimeBase,
};

pub fn register(reg: &mut CodecRegistry) {
    // WAVEFORMATEX tags handled by this crate:
    //   0x0001 WAVE_FORMAT_PCM — integer PCM, bit-depth disambiguation
    //     by bits_per_sample.
    //   0x0003 WAVE_FORMAT_IEEE_FLOAT — float PCM, same idea.
    let wf_int = CodecTag::wave_format(0x0001);
    let wf_flt = CodecTag::wave_format(0x0003);
    for (id, bits, tag, probe) in [
        (
            "pcm_u8",
            8u16,
            Some(&wf_int),
            probe_pcm_u8 as oxideav_core::ProbeFn,
        ),
        ("pcm_s8", 8, None, probe_pcm_s8 as oxideav_core::ProbeFn),
        (
            "pcm_s16le",
            16,
            Some(&wf_int),
            probe_pcm_s16le as oxideav_core::ProbeFn,
        ),
        (
            "pcm_s24le",
            24,
            Some(&wf_int),
            probe_pcm_s24le as oxideav_core::ProbeFn,
        ),
        (
            "pcm_s32le",
            32,
            Some(&wf_int),
            probe_pcm_s32le as oxideav_core::ProbeFn,
        ),
        (
            "pcm_f32le",
            32,
            Some(&wf_flt),
            probe_pcm_f32le as oxideav_core::ProbeFn,
        ),
        (
            "pcm_f64le",
            64,
            Some(&wf_flt),
            probe_pcm_f64le as oxideav_core::ProbeFn,
        ),
    ] {
        let _ = bits;
        let caps = CodecCapabilities::audio(format!("{id}_sw"))
            .with_lossless(true)
            .with_intra_only(true);
        let mut info = CodecInfo::new(CodecId::new(id))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder);
        if let Some(t) = tag {
            info = info.probe(probe).tag(t.clone());
        }
        reg.register(info);
    }

    for id in SLIN_ALIASES {
        let caps = CodecCapabilities::audio(format!("{id}_sw"))
            .with_lossless(true)
            .with_intra_only(true);
        // Same factories as pcm_s16le — `sample_format_for` maps all the
        // slin aliases to SampleFormat::S16 below. No WAVEFORMATEX claim.
        reg.register(
            CodecInfo::new(CodecId::new(*id))
                .capabilities(caps)
                .decoder(make_decoder)
                .encoder(make_encoder),
        );
    }
}

// --- Per-variant PCM probes -----------------------------------------------
// Each probe is selected by the WAVEFORMATEX tag at the call site; here we
// only need to disambiguate by bit depth. `bits_per_sample = None` returns
// 0.0 so the registry resolves nothing — the AVI demuxer then falls back
// to its static table which picks a sensible default.

fn match_bits(ctx: &ProbeContext, expected: u16) -> f32 {
    match ctx.bits_per_sample {
        Some(b) if b == expected => 1.0,
        _ => 0.0,
    }
}

fn probe_pcm_u8(ctx: &ProbeContext) -> f32 {
    match_bits(ctx, 8)
}
fn probe_pcm_s8(_ctx: &ProbeContext) -> f32 {
    // pcm_s8 has no canonical WAVEFORMATEX mapping; probe is unused but
    // required by the ProbeFn type alias in the register() table.
    0.0
}
fn probe_pcm_s16le(ctx: &ProbeContext) -> f32 {
    match ctx.bits_per_sample {
        // bits_per_sample == 0 occasionally means "unspecified" in the
        // wild — WAV files tagged as PCM that omit the depth. Treat
        // that as s16le (the most common default) with middling
        // confidence so a specific claimant (e.g. pcm_s24le with
        // bits=24) still wins if the depth is actually set.
        Some(0) => 0.5,
        Some(16) => 1.0,
        _ => 0.0,
    }
}
fn probe_pcm_s24le(ctx: &ProbeContext) -> f32 {
    match_bits(ctx, 24)
}
fn probe_pcm_s32le(ctx: &ProbeContext) -> f32 {
    match_bits(ctx, 32)
}
fn probe_pcm_f32le(ctx: &ProbeContext) -> f32 {
    match ctx.bits_per_sample {
        // IEEE float defaults to f32 when depth unspecified.
        None | Some(0) => 0.5,
        Some(32) => 1.0,
        _ => 0.0,
    }
}
fn probe_pcm_f64le(ctx: &ProbeContext) -> f32 {
    match_bits(ctx, 64)
}

/// Asterisk "signed linear" codec-id aliases. All are S16LE; the trailing
/// digits only matter at the container layer (see `slin.rs`).
pub(crate) const SLIN_ALIASES: &[&str] = &[
    "slin", "slin8", "slin16", "slin24", "slin32", "slin44", "slin48", "slin96", "slin192",
];

/// Return the [`SampleFormat`] implied by a PCM codec ID.
///
/// Also accepts the Asterisk `slin*` aliases, all of which describe 16-bit
/// signed linear PCM.
pub fn sample_format_for(id: &CodecId) -> Option<SampleFormat> {
    let s = id.as_str();
    Some(match s {
        "pcm_u8" => SampleFormat::U8,
        "pcm_s8" => SampleFormat::S8,
        "pcm_s16le" => SampleFormat::S16,
        "pcm_s24le" => SampleFormat::S24,
        "pcm_s32le" => SampleFormat::S32,
        "pcm_f32le" => SampleFormat::F32,
        "pcm_f64le" => SampleFormat::F64,
        _ if SLIN_ALIASES.contains(&s) => SampleFormat::S16,
        _ => return None,
    })
}

/// Return the canonical PCM codec ID for a [`SampleFormat`]. Planar formats
/// have no direct PCM codec — the caller must convert to interleaved first.
pub fn codec_id_for(fmt: SampleFormat) -> Option<CodecId> {
    Some(CodecId::new(match fmt {
        SampleFormat::U8 => "pcm_u8",
        SampleFormat::S8 => "pcm_s8",
        SampleFormat::S16 => "pcm_s16le",
        SampleFormat::S24 => "pcm_s24le",
        SampleFormat::S32 => "pcm_s32le",
        SampleFormat::F32 => "pcm_f32le",
        SampleFormat::F64 => "pcm_f64le",
        _ => return None,
    }))
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let format = sample_format_for(&params.codec_id)
        .ok_or_else(|| Error::CodecNotFound(params.codec_id.to_string()))?;
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("PCM decoder requires channels"))?;
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("PCM decoder requires sample_rate"))?;
    Ok(Box::new(PcmDecoder {
        id: params.codec_id.clone(),
        format,
        channels,
        sample_rate,
        pending: None,
        eof: false,
    }))
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let format = sample_format_for(&params.codec_id)
        .ok_or_else(|| Error::CodecNotFound(params.codec_id.to_string()))?;
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("PCM encoder requires channels"))?;
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("PCM encoder requires sample_rate"))?;
    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(format);
    Ok(Box::new(PcmEncoder {
        format,
        channels,
        sample_rate,
        output,
        queue: std::collections::VecDeque::new(),
    }))
}

struct PcmDecoder {
    id: CodecId,
    format: SampleFormat,
    channels: u16,
    sample_rate: u32,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for PcmDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "PCM decoder already has a buffered packet; call receive_frame first",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let bps = self.format.bytes_per_sample();
        let block = bps * self.channels as usize;
        if block == 0 || pkt.data.len() % block != 0 {
            return Err(Error::invalid("PCM packet size not a multiple of block"));
        }
        let samples = (pkt.data.len() / block) as u32;
        Ok(Frame::Audio(AudioFrame {
            format: self.format,
            channels: self.channels,
            sample_rate: self.sample_rate,
            samples,
            pts: pkt.pts,
            time_base: pkt.time_base,
            data: vec![pkt.data],
        }))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

struct PcmEncoder {
    format: SampleFormat,
    channels: u16,
    sample_rate: u32,
    output: CodecParameters,
    queue: std::collections::VecDeque<Packet>,
}

impl Encoder for PcmEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Audio(a) = frame else {
            return Err(Error::invalid("PCM encoder requires audio frames"));
        };
        if a.format != self.format
            || a.channels != self.channels
            || a.sample_rate != self.sample_rate
        {
            return Err(Error::invalid(
                "PCM encoder frame parameters do not match encoder configuration",
            ));
        }
        if a.format.is_planar() {
            return Err(Error::unsupported(
                "PCM encoder takes interleaved input; convert planar → interleaved first",
            ));
        }
        let data = a
            .data
            .first()
            .ok_or_else(|| Error::invalid("empty audio frame"))?
            .clone();
        let bps = a.format.bytes_per_sample() * a.channels as usize;
        let expected = bps * a.samples as usize;
        if data.len() != expected {
            return Err(Error::invalid("audio frame data length mismatch"));
        }
        let mut pkt = Packet::new(0, a.time_base, data);
        pkt.pts = a.pts;
        pkt.dts = a.pts;
        pkt.duration = Some(a.samples as i64);
        pkt.flags.keyframe = true;
        self.queue.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.queue.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Helper to build codec parameters for a PCM stream.
pub fn params(format: SampleFormat, channels: u16, sample_rate: u32) -> Result<CodecParameters> {
    let codec_id = codec_id_for(format)
        .ok_or_else(|| Error::unsupported(format!("no PCM codec for {:?}", format)))?;
    let mut p = CodecParameters::audio(codec_id);
    p.sample_format = Some(format);
    p.channels = Some(channels);
    p.sample_rate = Some(sample_rate);
    p.bit_rate =
        Some((format.bytes_per_sample() as u64) * 8 * (channels as u64) * (sample_rate as u64));
    Ok(p)
}

/// Default time base for a PCM audio stream: 1 / sample_rate.
pub fn time_base_for(sample_rate: u32) -> TimeBase {
    TimeBase::new(1, sample_rate as i64)
}
