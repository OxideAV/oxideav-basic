//! PCM codec: uncompressed interleaved linear PCM.
//!
//! A PCM "codec" is trivial — the packet payload *is* the sample data,
//! up to byte order and signedness. We still funnel it through
//! [`Decoder`] / [`Encoder`] so that pipelines treat it uniformly.
//!
//! # Codec ids and their decoded sample format
//!
//! The decoders always emit one of the core interleaved
//! [`SampleFormat`]s (little-endian, two's-complement for integers,
//! offset binary only for 8-bit); the encoders take that same format
//! and write the codec's wire layout.
//!
//! | codec id | wire layout | decoded / encoder input |
//! |---|---|---|
//! | `pcm_u8` | unsigned 8-bit (offset binary, 128 = silence) | `U8` |
//! | `pcm_s8` | signed 8-bit | `S8` |
//! | `pcm_s16le` / `pcm_s16be` | signed 16-bit LE / BE | `S16` |
//! | `pcm_u16le` / `pcm_u16be` | unsigned 16-bit LE / BE | `S16` |
//! | `pcm_s24le` / `pcm_s24be` | signed 24-bit packed LE / BE | `S24` |
//! | `pcm_u24le` / `pcm_u24be` | unsigned 24-bit packed LE / BE | `S24` |
//! | `pcm_s32le` / `pcm_s32be` | signed 32-bit LE / BE | `S32` |
//! | `pcm_u32le` / `pcm_u32be` | unsigned 32-bit LE / BE | `S32` |
//! | `pcm_f32le` / `pcm_f32be` | IEEE 754 binary32 LE / BE | `F32` |
//! | `pcm_f64le` / `pcm_f64be` | IEEE 754 binary64 LE / BE | `F64` |
//!
//! The big-endian ids are what QuickTime `twos` / `in24` / `in32` /
//! `fl32` / `fl64` and `lpcm` with `kAudioFormatFlagIsBigEndian`
//! (QTFF, "Sound Sample Description (Version 2)", LPCM flag values),
//! AIFF, Matroska `A_PCM/INT/BIG` and MPEG LPCM carry. Unsigned wide
//! samples (`lpcm` without `kAudioFormatFlagIsSignedInteger`) are
//! offset binary: the decoded two's-complement value is the wire value
//! with its most significant bit inverted, which is lossless both ways.
//!
//! 64-bit integer PCM (`pcm_s64le` / `pcm_s64be`) is not registered:
//! the core sample-format set has no 64-bit integer format to decode
//! it into losslessly. G.711 A-law / µ-law are `oxideav-g711`'s.
//!
//! Asterisk-style signed-linear aliases:
//! - `slin`, `slin8`, `slin16`, `slin24`, `slin32`, `slin44`, `slin48`,
//!   `slin96`, `slin192` — all map onto the `pcm_s16le` implementation.
//!   The trailing digits only indicate the implied sample rate of the
//!   surrounding headerless `.sln*` container (see `slin.rs`); as a codec
//!   they are indistinguishable from `pcm_s16le`.
//!
//! # Stream parameters and mid-stream changes
//!
//! A PCM packet carries no header, so a decoder cannot discover a new
//! shape on its own: [`Decoder::output_params`] states the codec's
//! decoded sample format with the stream's channels / rate / layout
//! from construction on, and the generation stays `0` — no
//! [`Frame::StreamParams`] event is ever emitted.
//!
//! The encoders follow a [`Frame::StreamParams`] event that changes the
//! **sample rate**, **channel count** or **channel layout**: every
//! sample is still self-contained, so later packets simply carry the
//! new shape (and a `1 / new_rate` time base), and
//! [`Encoder::output_params`] describes it from then on. An event that
//! changes the **sample format** (the encoder's codec id is fixed to
//! one wire layout), names a non-audio stream, or states zero channels
//! / a zero rate is rejected with [`Error::Unsupported`] and leaves the
//! encoder unchanged.

use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecParameters, CodecTag, Error, Frame, MediaType,
    Packet, ProbeContext, Result, SampleFormat, TimeBase,
};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder, Encoder};

/// Wire layout of one PCM codec id: how the bytes of one sample are
/// stored in a packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PcmLayout {
    /// Bits per sample on the wire (8, 16, 24, 32 or 64).
    pub bits: u16,
    /// IEEE 754 floating point (binary32 / binary64).
    pub float: bool,
    /// Two's-complement integers; `false` is offset binary. Always
    /// `true` for float layouts.
    pub signed: bool,
    /// Most significant byte first. Always `false` for 8-bit layouts.
    pub big_endian: bool,
}

impl PcmLayout {
    /// A layout from its parts.
    pub fn new(bits: u16, float: bool, signed: bool, big_endian: bool) -> Self {
        Self {
            bits,
            float,
            signed: signed || float,
            big_endian: big_endian && bits > 8,
        }
    }

    /// Bytes per sample on the wire.
    pub fn bytes_per_sample(&self) -> usize {
        self.bits as usize / 8
    }

    /// The core interleaved [`SampleFormat`] this layout decodes to,
    /// `None` when core has none that holds it losslessly.
    pub fn decoded_sample_format(&self) -> Option<SampleFormat> {
        Some(match (self.float, self.signed, self.bits) {
            (true, _, 32) => SampleFormat::F32,
            (true, _, 64) => SampleFormat::F64,
            (false, false, 8) => SampleFormat::U8,
            (false, true, 8) => SampleFormat::S8,
            (false, _, 16) => SampleFormat::S16,
            (false, _, 24) => SampleFormat::S24,
            (false, _, 32) => SampleFormat::S32,
            _ => return None,
        })
    }

    /// `true` when the wire bytes already are the decoded
    /// [`SampleFormat`]'s bytes (little-endian, and two's-complement
    /// for every width above 8 bits) — the layouts a container that
    /// stores core samples verbatim (WAV, `.sln`) can write directly.
    pub fn is_native(&self) -> bool {
        !self.big_endian && (self.signed || self.bits == 8)
    }

    /// `true` when decoding needs a byte-level transform (a byte swap
    /// and / or an inverted sign bit).
    fn needs_transform(&self) -> bool {
        !self.is_native()
    }
}

/// One registered PCM codec id.
struct Variant {
    id: &'static str,
    layout: PcmLayout,
}

const fn v(id: &'static str, bits: u16, float: bool, signed: bool, big_endian: bool) -> Variant {
    Variant {
        id,
        layout: PcmLayout {
            bits,
            float,
            signed,
            big_endian,
        },
    }
}

/// Every PCM codec id this crate decodes and encodes.
const VARIANTS: &[Variant] = &[
    v("pcm_u8", 8, false, false, false),
    v("pcm_s8", 8, false, true, false),
    v("pcm_s16le", 16, false, true, false),
    v("pcm_s16be", 16, false, true, true),
    v("pcm_u16le", 16, false, false, false),
    v("pcm_u16be", 16, false, false, true),
    v("pcm_s24le", 24, false, true, false),
    v("pcm_s24be", 24, false, true, true),
    v("pcm_u24le", 24, false, false, false),
    v("pcm_u24be", 24, false, false, true),
    v("pcm_s32le", 32, false, true, false),
    v("pcm_s32be", 32, false, true, true),
    v("pcm_u32le", 32, false, false, false),
    v("pcm_u32be", 32, false, false, true),
    v("pcm_f32le", 32, true, true, false),
    v("pcm_f32be", 32, true, true, true),
    v("pcm_f64le", 64, true, true, false),
    v("pcm_f64be", 64, true, true, true),
];

/// The codec ids [`register`] installs (excluding the `slin*` aliases).
pub fn codec_ids() -> impl Iterator<Item = &'static str> {
    VARIANTS.iter().map(|v| v.id)
}

pub fn register(reg: &mut CodecRegistry) {
    // WAVEFORMATEX tags handled by this crate:
    //   0x0001 WAVE_FORMAT_PCM — integer PCM, bit-depth disambiguation
    //     by bits_per_sample.
    //   0x0003 WAVE_FORMAT_IEEE_FLOAT — float PCM, same idea.
    // WAVE data is little-endian (two's-complement above 8 bits), so
    // only the native layouts claim a tag.
    let wf_int = CodecTag::wave_format(0x0001);
    let wf_flt = CodecTag::wave_format(0x0003);
    for var in VARIANTS {
        let fmt = var
            .layout
            .decoded_sample_format()
            .expect("every registered layout decodes to a core format");
        // The encoder takes the one interleaved core layout; pipelines
        // convert other layouts (planar, other widths) in front of it.
        let caps = CodecCapabilities::audio(format!("{}_sw", var.id))
            .with_lossless(true)
            .with_intra_only(true)
            .with_sample_formats(vec![fmt]);
        let mut info = CodecInfo::new(CodecId::new(var.id))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder);
        let probe: Option<(oxideav_core::ProbeFn, &CodecTag)> = match var.id {
            "pcm_u8" => Some((probe_pcm_u8, &wf_int)),
            "pcm_s16le" => Some((probe_pcm_s16le, &wf_int)),
            "pcm_s24le" => Some((probe_pcm_s24le, &wf_int)),
            "pcm_s32le" => Some((probe_pcm_s32le, &wf_int)),
            "pcm_f32le" => Some((probe_pcm_f32le, &wf_flt)),
            "pcm_f64le" => Some((probe_pcm_f64le, &wf_flt)),
            _ => None,
        };
        if let Some((probe, tag)) = probe {
            info = info.probe(probe).tag(tag.clone());
        }
        reg.register(info);
    }

    for id in SLIN_ALIASES {
        let caps = CodecCapabilities::audio(format!("{id}_sw"))
            .with_lossless(true)
            .with_intra_only(true)
            .with_sample_formats(vec![SampleFormat::S16]);
        // Same factories as pcm_s16le — `layout_for` maps all the slin
        // aliases to signed 16-bit LE below. No WAVEFORMATEX claim.
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

/// The wire layout of a PCM codec id this crate handles (including the
/// `slin*` aliases, which are signed 16-bit little-endian), `None` for
/// any other id.
pub fn layout_for(id: &CodecId) -> Option<PcmLayout> {
    let s = id.as_str();
    if SLIN_ALIASES.contains(&s) {
        return Some(PcmLayout::new(16, false, true, false));
    }
    VARIANTS.iter().find(|v| v.id == s).map(|v| v.layout)
}

/// Return the [`SampleFormat`] a PCM codec ID **decodes to** (and its
/// encoder takes) — for `pcm_s16be` that is `S16`, the core
/// little-endian format, not the wire layout. Use [`layout_for`] for
/// the wire layout, or [`native_sample_format_for`] when the wire bytes
/// must already be in the core format.
///
/// Also accepts the Asterisk `slin*` aliases, all of which describe 16-bit
/// signed linear PCM.
pub fn sample_format_for(id: &CodecId) -> Option<SampleFormat> {
    layout_for(id).and_then(|l| l.decoded_sample_format())
}

/// The [`SampleFormat`] whose bytes a PCM codec ID's packets **are**,
/// `None` when the id is unknown or its wire layout is not a core
/// format byte for byte (big-endian, unsigned above 8 bits). This is
/// the question a container that stores core samples verbatim asks.
pub fn native_sample_format_for(id: &CodecId) -> Option<SampleFormat> {
    layout_for(id)
        .filter(PcmLayout::is_native)
        .and_then(|l| l.decoded_sample_format())
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

/// Rewrite wire samples into the decoded core layout, in place.
/// `data.len()` must be a multiple of the sample size.
fn wire_to_core(layout: PcmLayout, data: &mut [u8]) {
    let n = layout.bytes_per_sample();
    let flip = !layout.signed && n > 1;
    for s in data.chunks_exact_mut(n) {
        if layout.big_endian {
            s.reverse();
        }
        if flip {
            s[n - 1] ^= 0x80;
        }
    }
}

/// Inverse of [`wire_to_core`], in place.
fn core_to_wire(layout: PcmLayout, data: &mut [u8]) {
    let n = layout.bytes_per_sample();
    let flip = !layout.signed && n > 1;
    for s in data.chunks_exact_mut(n) {
        if flip {
            s[n - 1] ^= 0x80;
        }
        if layout.big_endian {
            s.reverse();
        }
    }
}

/// Resolve the codec id of `params` and check the shape every PCM
/// codec needs: a known id, at least one channel, a non-zero rate.
fn resolve(params: &CodecParameters, who: &str) -> Result<(PcmLayout, SampleFormat, u16, u32)> {
    let layout = layout_for(&params.codec_id)
        .ok_or_else(|| Error::CodecNotFound(params.codec_id.to_string()))?;
    let format = layout
        .decoded_sample_format()
        .ok_or_else(|| Error::CodecNotFound(params.codec_id.to_string()))?;
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid(format!("PCM {who} requires channels")))?;
    if channels == 0 {
        return Err(Error::invalid(format!("PCM {who}: zero channels")));
    }
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid(format!("PCM {who} requires sample_rate")))?;
    if sample_rate == 0 {
        return Err(Error::invalid(format!("PCM {who}: zero sample_rate")));
    }
    Ok((layout, format, channels, sample_rate))
}

/// The bit rate of an uncompressed stream, saturating.
fn pcm_bit_rate(layout: PcmLayout, channels: u16, sample_rate: u32) -> u64 {
    (layout.bits as u64)
        .saturating_mul(channels as u64)
        .saturating_mul(sample_rate as u64)
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let (layout, format, channels, sample_rate) = resolve(params, "decoder")?;
    let mut output = CodecParameters::audio(params.codec_id.clone());
    output.sample_format = Some(format);
    output.channels = Some(channels);
    output.sample_rate = Some(sample_rate);
    output.channel_layout = params
        .channel_layout
        .filter(|l| l.channel_count() == channels);
    Ok(Box::new(PcmDecoder {
        layout,
        block: layout.bytes_per_sample() * channels as usize,
        output,
        pending: None,
        eof: false,
    }))
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let (layout, format, channels, sample_rate) = resolve(params, "encoder")?;
    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(format);
    Ok(Box::new(PcmEncoder {
        layout,
        format,
        channels,
        sample_rate,
        output,
        queue: std::collections::VecDeque::new(),
    }))
}

struct PcmDecoder {
    layout: PcmLayout,
    /// Bytes per interleaved sample frame (all channels).
    block: usize,
    output: CodecParameters,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for PcmDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.output.codec_id
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
        if pkt.data.len() % self.block != 0 {
            return Err(Error::invalid("PCM packet size not a multiple of block"));
        }
        let samples = u32::try_from(pkt.data.len() / self.block)
            .map_err(|_| Error::invalid("PCM packet holds more than 2^32 samples"))?;
        let mut data = pkt.data;
        if self.layout.needs_transform() {
            wire_to_core(self.layout, &mut data);
        }
        Ok(Frame::Audio(AudioFrame {
            samples,
            pts: pkt.pts,
            data: vec![data],
        }))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // A seek resumes decoding: drop the buffered packet and leave
        // the end-of-stream state, so the next empty poll answers
        // NeedMore rather than Eof.
        self.pending = None;
        self.eof = false;
        Ok(())
    }

    fn output_params(&self) -> Option<CodecParameters> {
        Some(self.output.clone())
    }
}

struct PcmEncoder {
    layout: PcmLayout,
    format: SampleFormat,
    channels: u16,
    sample_rate: u32,
    output: CodecParameters,
    queue: std::collections::VecDeque<Packet>,
}

impl PcmEncoder {
    /// Follow a mid-stream input-shape change (see the module docs for
    /// what is followed and what is refused).
    fn reconfigure(&mut self, p: &CodecParameters) -> Result<()> {
        if p.media_type != MediaType::Audio {
            return Err(Error::unsupported(
                "PCM encoder: stream-params event for a non-audio stream",
            ));
        }
        if let Some(f) = p.sample_format
            && f != self.format
        {
            return Err(Error::unsupported(format!(
                "PCM encoder for {} takes {:?}; a switch to {:?} needs a converter in front",
                self.output.codec_id, self.format, f
            )));
        }
        let channels = p
            .channels
            .or_else(|| p.channel_layout.map(|l| l.channel_count()))
            .unwrap_or(self.channels);
        let sample_rate = p.sample_rate.unwrap_or(self.sample_rate);
        if channels == 0 || sample_rate == 0 {
            return Err(Error::unsupported(
                "PCM encoder: cannot encode zero channels or a zero sample rate",
            ));
        }
        let layout = match p.channel_layout {
            Some(l) if l.channel_count() == channels => Some(l),
            Some(_) => None,
            None => self
                .output
                .channel_layout
                .filter(|l| l.channel_count() == channels),
        };
        if (channels != self.channels || sample_rate != self.sample_rate)
            && self.output.bit_rate.is_some()
        {
            self.output.bit_rate = Some(pcm_bit_rate(self.layout, channels, sample_rate));
        }
        self.channels = channels;
        self.sample_rate = sample_rate;
        self.output.channels = Some(channels);
        self.output.sample_rate = Some(sample_rate);
        self.output.channel_layout = layout;
        Ok(())
    }
}

impl Encoder for PcmEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let a = match frame {
            Frame::Audio(a) => a,
            Frame::StreamParams(p) => return self.reconfigure(p),
            _ => return Err(Error::invalid("PCM encoder requires audio frames")),
        };
        // Per-frame format/channels/sample_rate are not carried on
        // AudioFrame — the encoder enforces its configured layout via the
        // stored fields and the byte-count check below.
        let plane = a
            .data
            .first()
            .ok_or_else(|| Error::invalid("empty audio frame"))?;
        let expected = (self.format.bytes_per_sample() * self.channels as usize)
            .checked_mul(a.samples as usize)
            .ok_or_else(|| Error::invalid("audio frame size overflow"))?;
        if a.data.len() != 1 || plane.len() != expected {
            return Err(Error::invalid("audio frame data length mismatch"));
        }
        let mut data = plane.clone();
        if self.layout.needs_transform() {
            core_to_wire(self.layout, &mut data);
        }
        let mut pkt = Packet::new(0, TimeBase::new(1, self.sample_rate as i64), data);
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

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::ChannelLayout;

    fn reg() -> CodecRegistry {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        reg
    }

    fn stream(id: &str, channels: u16, rate: u32) -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new(id));
        p.channels = Some(channels);
        p.sample_rate = Some(rate);
        p
    }

    #[test]
    fn encoders_declare_their_interleaved_layout() {
        let reg = reg();
        for (id, fmt) in [
            ("pcm_s16le", SampleFormat::S16),
            ("pcm_s16be", SampleFormat::S16),
            ("pcm_u24be", SampleFormat::S24),
            ("pcm_f32le", SampleFormat::F32),
            ("pcm_f64be", SampleFormat::F64),
            ("pcm_u8", SampleFormat::U8),
        ] {
            let imp = &reg.implementations(&CodecId::new(id))[0];
            assert_eq!(imp.caps.accepted_sample_formats, vec![fmt], "{id}");
        }
    }

    #[test]
    fn every_variant_has_a_core_format_and_round_trips() {
        // One sample frame of every width, distinct bytes.
        for var in VARIANTS {
            let n = var.layout.bytes_per_sample();
            let wire: Vec<u8> = (0..n as u8 * 3)
                .map(|i| i.wrapping_mul(37) ^ 0x5a)
                .collect();
            let mut core = wire.clone();
            wire_to_core(var.layout, &mut core);
            let mut back = core.clone();
            core_to_wire(var.layout, &mut back);
            assert_eq!(back, wire, "{}", var.id);
            assert!(var.layout.decoded_sample_format().is_some(), "{}", var.id);
        }
    }

    #[test]
    fn wire_transforms_match_hand_values() {
        // -2 as s16 big-endian → LE core bytes.
        let mut d = vec![0xFF, 0xFE];
        wire_to_core(layout_for(&CodecId::new("pcm_s16be")).unwrap(), &mut d);
        assert_eq!(i16::from_le_bytes([d[0], d[1]]), -2);
        // Offset binary 0x8000 is silence → 0.
        let mut d = vec![0x00, 0x80];
        wire_to_core(layout_for(&CodecId::new("pcm_u16le")).unwrap(), &mut d);
        assert_eq!(d, vec![0, 0]);
        // u24be 0x000000 is the most negative value.
        let mut d = vec![0x00, 0x00, 0x00];
        wire_to_core(layout_for(&CodecId::new("pcm_u24be")).unwrap(), &mut d);
        assert_eq!(d, vec![0x00, 0x00, 0x80]);
        // f32 BE 1.0.
        let mut d = 1.0f32.to_be_bytes().to_vec();
        wire_to_core(layout_for(&CodecId::new("pcm_f32be")).unwrap(), &mut d);
        assert_eq!(f32::from_le_bytes([d[0], d[1], d[2], d[3]]), 1.0);
    }

    #[test]
    fn native_vs_decoded_sample_format() {
        let id = CodecId::new("pcm_s24be");
        assert_eq!(sample_format_for(&id), Some(SampleFormat::S24));
        assert_eq!(native_sample_format_for(&id), None);
        let id = CodecId::new("pcm_u8");
        assert_eq!(native_sample_format_for(&id), Some(SampleFormat::U8));
        assert_eq!(
            native_sample_format_for(&CodecId::new("slin16")),
            Some(SampleFormat::S16)
        );
        assert_eq!(sample_format_for(&CodecId::new("pcm_s64le")), None);
        assert_eq!(sample_format_for(&CodecId::new("pcm_alaw")), None);
    }

    #[test]
    fn decoder_states_output_params_and_never_moves_generation() {
        let reg = reg();
        let mut p = stream("pcm_s32be", 2, 48_000);
        p.sample_format = Some(SampleFormat::F32); // a wrong container claim
        p.channel_layout = Some(ChannelLayout::Stereo);
        let mut d = reg.first_decoder(&p).unwrap();
        let out = d.output_params().unwrap();
        assert_eq!(out.sample_format, Some(SampleFormat::S32));
        assert_eq!(out.channels, Some(2));
        assert_eq!(out.sample_rate, Some(48_000));
        assert_eq!(out.channel_layout, Some(ChannelLayout::Stereo));
        let pkt = Packet::new(0, TimeBase::new(1, 48_000), vec![0, 0, 0, 1, 0x80, 0, 0, 0]);
        d.send_packet(&pkt).unwrap();
        let Frame::Audio(a) = d.receive_frame().unwrap() else {
            panic!("audio frame expected")
        };
        assert_eq!(a.samples, 1);
        assert_eq!(a.data[0], vec![1, 0, 0, 0, 0, 0, 0, 0x80]);
        assert_eq!(d.output_params_generation(), 0);
    }

    #[test]
    fn decoder_layout_disagreeing_with_channels_is_dropped() {
        let mut p = stream("pcm_s16le", 1, 8_000);
        p.channel_layout = Some(ChannelLayout::Stereo);
        let d = make_decoder(&p).unwrap();
        assert_eq!(d.output_params().unwrap().channel_layout, None);
    }

    #[test]
    fn factories_reject_zero_shapes_and_unknown_ids() {
        assert!(make_decoder(&stream("pcm_s16le", 0, 8_000)).is_err());
        assert!(make_decoder(&stream("pcm_s16le", 1, 0)).is_err());
        assert!(make_encoder(&stream("pcm_s16le", 0, 8_000)).is_err());
        assert!(make_encoder(&stream("pcm_s64le", 1, 8_000)).is_err());
    }

    #[test]
    fn decoder_reset_resumes_after_flush() {
        let mut d = make_decoder(&stream("pcm_u8", 1, 8_000)).unwrap();
        d.flush().unwrap();
        assert!(matches!(d.receive_frame(), Err(Error::Eof)));
        d.reset().unwrap();
        assert!(matches!(d.receive_frame(), Err(Error::NeedMore)));
        d.send_packet(&Packet::new(0, TimeBase::new(1, 8_000), vec![128]))
            .unwrap();
        assert!(d.receive_frame().is_ok());
        assert!(matches!(d.receive_frame(), Err(Error::NeedMore)));
    }

    #[test]
    fn decoder_rejects_partial_sample_frame() {
        let mut d = make_decoder(&stream("pcm_s24be", 2, 8_000)).unwrap();
        d.send_packet(&Packet::new(0, TimeBase::new(1, 8_000), vec![0; 7]))
            .unwrap();
        assert!(matches!(d.receive_frame(), Err(Error::InvalidData(_))));
    }

    /// An interleaved S16 frame of `samples` spread over `channels`.
    fn s16_frame(samples: &[i16], channels: usize, pts: i64) -> Frame {
        Frame::Audio(AudioFrame {
            samples: (samples.len() / channels) as u32,
            pts: Some(pts),
            data: vec![samples.iter().flat_map(|s| s.to_le_bytes()).collect()],
        })
    }

    #[test]
    fn encoder_follows_rate_and_channel_changes() {
        let mut p = stream("pcm_s16be", 1, 8_000);
        p.bit_rate = Some(128_000);
        let mut e = make_encoder(&p).unwrap();
        e.send_frame(&s16_frame(&[1, -1], 1, 0)).unwrap();
        let pkt = e.receive_packet().unwrap();
        assert_eq!(pkt.data, vec![0, 1, 0xFF, 0xFF]);
        assert_eq!(pkt.time_base, TimeBase::new(1, 8_000));

        let mut ev = stream("pcm_s16be", 2, 16_000);
        ev.sample_format = Some(SampleFormat::S16);
        ev.channel_layout = Some(ChannelLayout::Stereo);
        e.send_frame(&Frame::StreamParams(Box::new(ev))).unwrap();
        let out = e.output_params();
        assert_eq!(out.channels, Some(2));
        assert_eq!(out.sample_rate, Some(16_000));
        assert_eq!(out.channel_layout, Some(ChannelLayout::Stereo));
        assert_eq!(out.bit_rate, Some(512_000));
        // Two stereo sample frames now.
        e.send_frame(&s16_frame(&[1, 2, 3, 4], 2, 2)).unwrap();
        let pkt = e.receive_packet().unwrap();
        assert_eq!(pkt.data.len(), 8);
        assert_eq!(pkt.duration, Some(2));
        assert_eq!(pkt.time_base, TimeBase::new(1, 16_000));
        // An old-shape frame is now a length mismatch.
        assert!(e.send_frame(&s16_frame(&[1], 1, 4)).is_err());
    }

    #[test]
    fn encoder_refuses_unfollowable_events_unchanged() {
        let mut e = make_encoder(&stream("pcm_f32le", 2, 44_100)).unwrap();
        let mut ev = stream("pcm_f32le", 2, 44_100);
        ev.sample_format = Some(SampleFormat::S16);
        let r = e.send_frame(&Frame::StreamParams(Box::new(ev)));
        assert!(matches!(r, Err(Error::Unsupported(_))), "{r:?}");
        let r = e.send_frame(&Frame::StreamParams(Box::new(stream(
            "pcm_f32le",
            0,
            44_100,
        ))));
        assert!(matches!(r, Err(Error::Unsupported(_))), "{r:?}");
        let mut video = CodecParameters::video(CodecId::new("rawvideo"));
        video.width = Some(2);
        let r = e.send_frame(&Frame::StreamParams(Box::new(video)));
        assert!(matches!(r, Err(Error::Unsupported(_))), "{r:?}");
        assert_eq!(e.output_params().channels, Some(2));
        assert_eq!(e.output_params().sample_rate, Some(44_100));
        // A same-shape event is a no-op.
        e.send_frame(&Frame::StreamParams(Box::new(stream(
            "pcm_f32le",
            2,
            44_100,
        ))))
        .unwrap();
        assert!(matches!(e.receive_packet(), Err(Error::NeedMore)));
    }

    #[test]
    fn encoder_rejects_hostile_frame_sizes() {
        let mut e = make_encoder(&stream("pcm_s16le", 2, 8_000)).unwrap();
        let huge = Frame::Audio(AudioFrame {
            samples: u32::MAX,
            pts: None,
            data: vec![vec![0; 4]],
        });
        assert!(e.send_frame(&huge).is_err());
        let two_planes = Frame::Audio(AudioFrame {
            samples: 1,
            pts: None,
            data: vec![vec![0; 4], vec![0; 4]],
        });
        assert!(e.send_frame(&two_planes).is_err());
    }
}
