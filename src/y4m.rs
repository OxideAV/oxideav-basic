//! YUV4MPEG2 (`.y4m`) raw-video container.
//!
//! A `.y4m` file is the simplest possible video container: a single
//! ASCII header line followed by a sequence of `FRAME\n`-prefixed raw
//! planar YUV pictures. There is no index, no random-access support,
//! and no in-stream codec tag — every payload is rawvideo.
//!
//! # Wire format
//!
//! ```text
//! YUV4MPEG2 W<w> H<h> F<num>:<den> Ip A<n>:<d> C<colorspace> [X<k>=<v>...]\n
//! FRAME [X<k>=<v>...]\n
//! <Y plane><U plane><V plane>
//! FRAME ...\n
//! ...
//! ```
//!
//! - The header begins with the literal `YUV4MPEG2 ` magic (10 bytes,
//!   trailing space included).
//! - Each parameter is a single ASCII letter followed by a value, with
//!   space-separated tokens and a `\n` terminator.
//! - `W` (width) and `H` (height) are required.
//! - `F<num>:<den>` is the frame rate.
//! - `I` is the interlace mode: `p` progressive (default), `t` top
//!   first, `b` bottom first, `m` mixed.
//! - `A<num>:<den>` is the pixel aspect ratio.
//! - `C<colorspace>` selects chroma subsampling and bit depth. Common
//!   values: `420jpeg`, `420mpeg2`, `420paldv`, `420`, `422`, `444`,
//!   `mono`, with optional `p10`/`p12`/`p14`/`p16` suffix for higher
//!   bit depths.
//! - `X<key>=<val>` extension tokens are preserved verbatim in the
//!   demuxer's [`metadata()`](Demuxer::metadata) output and ignored by
//!   the muxer (per-frame extensions are simply dropped on decode and
//!   never emitted on encode).
//!
//! # Plane sizes
//!
//! For an 8-bit stream:
//!
//! | colorspace | Y bytes | U bytes        | V bytes        |
//! |------------|---------|----------------|----------------|
//! | `420*`     | `w*h`   | `(w/2)*(h/2)`  | `(w/2)*(h/2)`  |
//! | `422`      | `w*h`   | `(w/2)*h`      | `(w/2)*h`      |
//! | `444`      | `w*h`   | `w*h`          | `w*h`          |
//! | `mono`     | `w*h`   | 0              | 0              |
//!
//! `p10`/`p12`/`p14`/`p16` doubles every byte count (16-bit
//! little-endian samples, per the ffmpeg convention).
//!
//! # Codec id
//!
//! Frames are emitted as packets carrying the `rawvideo` codec id.
//! Pixel format and dimensions are exposed via the stream's
//! [`CodecParameters`].

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, Error, MediaType, Packet, PixelFormat, Rational,
    Result, StreamInfo, TimeBase,
};
use oxideav_core::{ContainerRegistry, Demuxer, Muxer, ProbeData, ReadSeek, WriteSeek};
use std::io::{Read, Write};

/// 10-byte magic prefix (the trailing space is part of the magic).
pub(crate) const MAGIC: &[u8; 10] = b"YUV4MPEG2 ";

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("y4m", open_demuxer);
    reg.register_muxer("y4m", open_muxer);
    reg.register_extension("y4m", "y4m");
    reg.register_extension("yuv4mpeg", "y4m");
    reg.register_probe("y4m", probe);
}

/// `YUV4MPEG2 ` magic at offset 0 — unambiguous.
fn probe(p: &ProbeData) -> u8 {
    if p.buf.len() < MAGIC.len() {
        return 0;
    }
    if &p.buf[..MAGIC.len()] == MAGIC.as_slice() {
        100
    } else {
        0
    }
}

// ───────────────────────── header parser ─────────────────────────

/// Parsed header parameters.
#[derive(Clone, Debug)]
struct Y4mHeader {
    width: u32,
    height: u32,
    frame_rate: Rational,
    pixel_aspect: Rational,
    interlace: u8,
    colorspace: String,
    /// Extension parameters (`X<k>=<v>`) preserved as `("X<k>", "<v>")`
    /// pairs to round-trip them through `Demuxer::metadata`.
    x_params: Vec<(String, String)>,
}

/// Map a Y4M `C<colorspace>` token to a [`PixelFormat`] plus the chroma
/// subsampling (`hss`, `vss`) and bytes-per-sample. Returns `None` for
/// formats this implementation can't represent in `oxideav-core`.
fn pixel_format_for_colorspace(c: &str) -> Option<(PixelFormat, u32, u32, u32)> {
    // (pf, horizontal_chroma_shift, vertical_chroma_shift, bytes_per_sample)
    match c {
        // 8-bit limited-range (TV-range) variants — there is no
        // dedicated PixelFormat for paldv/mpeg2/jpeg colour matrices,
        // so they all map to Yuv420P. Demuxer::metadata preserves the
        // original tag for clients that care.
        "420" | "420jpeg" | "420mpeg2" | "420paldv" => Some((PixelFormat::Yuv420P, 1, 1, 1)),
        "422" => Some((PixelFormat::Yuv422P, 1, 0, 1)),
        "444" => Some((PixelFormat::Yuv444P, 0, 0, 1)),
        "mono" => Some((PixelFormat::Gray8, 0, 0, 1)),

        // 10/12-bit variants. The Y4M convention is little-endian per ffmpeg.
        "420p10" => Some((PixelFormat::Yuv420P10Le, 1, 1, 2)),
        "422p10" => Some((PixelFormat::Yuv422P10Le, 1, 0, 2)),
        "444p10" => Some((PixelFormat::Yuv444P10Le, 0, 0, 2)),
        "420p12" => Some((PixelFormat::Yuv420P12Le, 1, 1, 2)),
        "422p12" => Some((PixelFormat::Yuv422P12Le, 1, 0, 2)),
        "444p12" => Some((PixelFormat::Yuv444P12Le, 0, 0, 2)),
        "monop10" | "monop12" | "monop16" => Some((PixelFormat::Gray16Le, 0, 0, 2)),
        _ => None,
    }
}

/// Inverse of [`pixel_format_for_colorspace`] — pick a Y4M `C<...>`
/// token that the muxer can write for a given pixel format. Defaults
/// stay close to the ffmpeg convention (`420mpeg2` → `420mpeg2`,
/// generic `Yuv420P` → `420jpeg`).
fn colorspace_for_pixel_format(pf: PixelFormat) -> Result<&'static str> {
    Ok(match pf {
        PixelFormat::Yuv420P => "420mpeg2",
        PixelFormat::YuvJ420P => "420jpeg",
        PixelFormat::Yuv422P | PixelFormat::YuvJ422P => "422",
        PixelFormat::Yuv444P | PixelFormat::YuvJ444P => "444",
        PixelFormat::Gray8 => "mono",
        PixelFormat::Yuv420P10Le => "420p10",
        PixelFormat::Yuv422P10Le => "422p10",
        PixelFormat::Yuv444P10Le => "444p10",
        PixelFormat::Yuv420P12Le => "420p12",
        PixelFormat::Yuv422P12Le => "422p12",
        PixelFormat::Yuv444P12Le => "444p12",
        PixelFormat::Gray16Le => "monop16",
        other => {
            return Err(Error::unsupported(format!(
                "y4m muxer: pixel format {:?} cannot be expressed as a Y4M colorspace",
                other
            )));
        }
    })
}

/// Read a single line ending in `\n` from `input`. The terminating
/// newline is consumed but not included in the returned bytes. Returns
/// `Error::invalid` if the line exceeds `cap` bytes (DoS guard) or if
/// the stream ends before a newline arrives.
fn read_line(input: &mut dyn ReadSeek, cap: usize) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(64);
    let mut byte = [0u8; 1];
    loop {
        match input.read(&mut byte) {
            Ok(0) => {
                if buf.is_empty() {
                    return Err(Error::Eof);
                }
                return Err(Error::invalid("y4m: unterminated header/FRAME line"));
            }
            Ok(_) => {}
            Err(e) => return Err(Error::from(e)),
        }
        if byte[0] == b'\n' {
            return Ok(buf);
        }
        if buf.len() >= cap {
            return Err(Error::invalid("y4m: header/FRAME line exceeded sanity cap"));
        }
        buf.push(byte[0]);
    }
}

/// Parse a `<num>:<den>` ratio. Both numerator and denominator must be
/// non-negative `i64`s.
fn parse_ratio(s: &str, ctx: &str) -> Result<Rational> {
    let mut it = s.splitn(2, ':');
    let num = it
        .next()
        .ok_or_else(|| Error::invalid(format!("y4m: malformed {ctx} '{s}'")))?
        .parse::<i64>()
        .map_err(|_| Error::invalid(format!("y4m: non-numeric {ctx} '{s}'")))?;
    let den = it
        .next()
        .ok_or_else(|| Error::invalid(format!("y4m: malformed {ctx} '{s}'")))?
        .parse::<i64>()
        .map_err(|_| Error::invalid(format!("y4m: non-numeric {ctx} '{s}'")))?;
    if num < 0 || den < 0 {
        return Err(Error::invalid(format!("y4m: negative {ctx} '{s}'")));
    }
    Ok(Rational::new(num, den))
}

/// Parse the full ASCII header line (without the trailing `\n`). The
/// magic prefix `YUV4MPEG2 ` must already be present.
fn parse_header(line: &[u8]) -> Result<Y4mHeader> {
    if line.len() < MAGIC.len() || &line[..MAGIC.len()] != MAGIC.as_slice() {
        return Err(Error::invalid("y4m: missing YUV4MPEG2 magic"));
    }
    // Per the spec the header is plain ASCII; surface a clear error
    // rather than letting String::from_utf8 panic on the rare bogus
    // file that contains arbitrary bytes after the magic.
    let body = std::str::from_utf8(&line[MAGIC.len()..])
        .map_err(|_| Error::invalid("y4m: non-ASCII header"))?;

    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut frame_rate: Option<Rational> = None;
    let mut pixel_aspect: Option<Rational> = None;
    let mut interlace: u8 = b'p';
    let mut colorspace: Option<String> = None;
    let mut x_params: Vec<(String, String)> = Vec::new();

    for tok in body.split(' ') {
        if tok.is_empty() {
            continue;
        }
        let (tag, val) = (tok.as_bytes()[0], &tok[1..]);
        match tag {
            b'W' => {
                width = Some(
                    val.parse()
                        .map_err(|_| Error::invalid(format!("y4m: bad width '{val}'")))?,
                );
            }
            b'H' => {
                height = Some(
                    val.parse()
                        .map_err(|_| Error::invalid(format!("y4m: bad height '{val}'")))?,
                );
            }
            b'F' => frame_rate = Some(parse_ratio(val, "frame rate")?),
            b'A' => pixel_aspect = Some(parse_ratio(val, "pixel aspect")?),
            b'I' => {
                interlace = *val.as_bytes().first().unwrap_or(&b'p');
                if !matches!(interlace, b'p' | b't' | b'b' | b'm' | b'?') {
                    return Err(Error::invalid(format!(
                        "y4m: unknown interlace mode '{val}'"
                    )));
                }
            }
            b'C' => colorspace = Some(val.to_string()),
            b'X' => {
                // Preserve the full token (key + '=' + value, or just
                // a key) as ("X<key>", "<value-or-empty>") in metadata.
                let (k, v) = match val.find('=') {
                    Some(i) => (&val[..i], &val[i + 1..]),
                    None => (val, ""),
                };
                x_params.push((format!("X{k}"), v.to_string()));
            }
            _ => {
                return Err(Error::invalid(format!(
                    "y4m: unknown header tag '{}' in '{tok}'",
                    tag as char
                )));
            }
        }
    }

    let width = width.ok_or_else(|| Error::invalid("y4m: header missing W (width)"))?;
    let height = height.ok_or_else(|| Error::invalid("y4m: header missing H (height)"))?;
    if width == 0 || height == 0 {
        return Err(Error::invalid("y4m: zero width or height"));
    }
    let frame_rate = frame_rate.unwrap_or(Rational::new(25, 1));
    let pixel_aspect = pixel_aspect.unwrap_or(Rational::new(0, 0));
    // Y4M's historic default when C is absent is 420jpeg.
    let colorspace = colorspace.unwrap_or_else(|| "420jpeg".to_string());

    Ok(Y4mHeader {
        width,
        height,
        frame_rate,
        pixel_aspect,
        interlace,
        colorspace,
        x_params,
    })
}

/// Total bytes per decoded frame for the given header.
fn frame_size(hdr: &Y4mHeader) -> Result<usize> {
    let (_, hss, vss, bps) = pixel_format_for_colorspace(&hdr.colorspace).ok_or_else(|| {
        Error::unsupported(format!(
            "y4m: colorspace '{}' not supported",
            hdr.colorspace
        ))
    })?;
    let w = hdr.width as usize;
    let h = hdr.height as usize;
    let bps = bps as usize;
    let y_bytes = w * h * bps;
    if hdr.colorspace.starts_with("mono") {
        return Ok(y_bytes);
    }
    let cw = w >> hss;
    let ch = h >> vss;
    let chroma = cw * ch * bps;
    Ok(y_bytes + 2 * chroma)
}

// ───────────────────────── Demuxer ─────────────────────────

/// Largest header / FRAME line we'll tolerate. Real-world headers are
/// well under 1 KiB; this cap keeps a malicious or truncated file from
/// burning unbounded memory in `read_line`.
const LINE_CAP: usize = 16 * 1024;

fn open_demuxer(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> Result<Box<dyn Demuxer>> {
    let line = read_line(&mut *input, LINE_CAP)?;
    let hdr = parse_header(&line)?;
    let pf_info = pixel_format_for_colorspace(&hdr.colorspace).ok_or_else(|| {
        Error::unsupported(format!(
            "y4m: colorspace '{}' not supported",
            hdr.colorspace
        ))
    })?;
    let (pix_fmt, _hss, _vss, _bps) = pf_info;
    let fsize = frame_size(&hdr)?;

    // Time base = inverse of frame rate. Defensive against `F0:0`.
    let time_base = if hdr.frame_rate.num > 0 && hdr.frame_rate.den > 0 {
        TimeBase::new(hdr.frame_rate.den, hdr.frame_rate.num)
    } else {
        TimeBase::new(1, 25)
    };

    let mut params = CodecParameters::video(CodecId::new("rawvideo"));
    params.width = Some(hdr.width);
    params.height = Some(hdr.height);
    params.pixel_format = Some(pix_fmt);
    params.frame_rate = Some(hdr.frame_rate);

    // Surface the parsed header in metadata so callers can inspect
    // colorspace / aspect / X-params without re-parsing.
    let mut metadata: Vec<(String, String)> = Vec::with_capacity(4 + hdr.x_params.len());
    metadata.push(("colorspace".into(), hdr.colorspace.clone()));
    metadata.push(("interlace".into(), (hdr.interlace as char).to_string()));
    metadata.push((
        "pixel_aspect".into(),
        format!("{}:{}", hdr.pixel_aspect.num, hdr.pixel_aspect.den),
    ));
    metadata.push((
        "frame_rate".into(),
        format!("{}:{}", hdr.frame_rate.num, hdr.frame_rate.den),
    ));
    metadata.extend(hdr.x_params);

    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: None,
        start_time: Some(0),
        params,
    };

    Ok(Box::new(Y4mDemuxer {
        input,
        streams: vec![stream],
        frame_size: fsize,
        frames_emitted: 0,
        metadata,
    }))
}

struct Y4mDemuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    frame_size: usize,
    frames_emitted: i64,
    metadata: Vec<(String, String)>,
}

impl Demuxer for Y4mDemuxer {
    fn format_name(&self) -> &str {
        "y4m"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        // Each frame begins with `FRAME[ X...]\n`. Treat a clean EOF
        // (no bytes left) as the end of stream.
        let line = match read_line(&mut *self.input, LINE_CAP) {
            Ok(v) => v,
            Err(Error::Eof) => return Err(Error::Eof),
            Err(e) => return Err(e),
        };
        if line.len() < 5 || &line[..5] != b"FRAME" {
            return Err(Error::invalid(
                "y4m: expected 'FRAME' marker between frames",
            ));
        }
        // Per-frame X-params after FRAME are valid but not surfaced —
        // the spec uses them for things like film-pulldown flags that
        // a generic rawvideo consumer cannot act on. They are still
        // syntactically tolerated.

        let mut buf = vec![0u8; self.frame_size];
        self.input.read_exact(&mut buf)?;
        let stream = &self.streams[0];
        let pts = self.frames_emitted;
        self.frames_emitted += 1;

        let mut pkt = Packet::new(0, stream.time_base, buf);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(1);
        pkt.flags.keyframe = true;
        Ok(pkt)
    }

    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }
}

// ───────────────────────── Muxer ─────────────────────────

fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::unsupported("y4m supports exactly one video stream"));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Video {
        return Err(Error::invalid("y4m stream must be video"));
    }
    let width = s
        .params
        .width
        .ok_or_else(|| Error::invalid("y4m muxer: missing width"))?;
    let height = s
        .params
        .height
        .ok_or_else(|| Error::invalid("y4m muxer: missing height"))?;
    let pix_fmt = s
        .params
        .pixel_format
        .ok_or_else(|| Error::invalid("y4m muxer: missing pixel_format"))?;
    let colorspace = colorspace_for_pixel_format(pix_fmt)?;
    // Frame rate falls back to 25/1 when unset — same default as the
    // demuxer chooses for a missing F<num>:<den> token.
    let frame_rate = s.params.frame_rate.unwrap_or(Rational::new(25, 1));
    if frame_rate.num <= 0 || frame_rate.den <= 0 {
        return Err(Error::invalid("y4m muxer: invalid frame_rate"));
    }

    let (_, hss, vss, bps) = pixel_format_for_colorspace(colorspace).ok_or_else(|| {
        Error::other(format!(
            "y4m muxer: internal colorspace lookup failed for '{colorspace}'"
        ))
    })?;
    let w = width as usize;
    let h = height as usize;
    let bps = bps as usize;
    let chroma = if colorspace.starts_with("mono") {
        0
    } else {
        let cw = w >> hss;
        let ch = h >> vss;
        cw * ch * bps
    };
    let frame_size = w * h * bps + 2 * chroma;

    Ok(Box::new(Y4mMuxer {
        output,
        width,
        height,
        frame_rate,
        colorspace,
        frame_size,
        header_written: false,
        trailer_written: false,
    }))
}

struct Y4mMuxer {
    output: Box<dyn WriteSeek>,
    width: u32,
    height: u32,
    frame_rate: Rational,
    colorspace: &'static str,
    frame_size: usize,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for Y4mMuxer {
    fn format_name(&self) -> &str {
        "y4m"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("y4m header already written"));
        }
        // YUV4MPEG2 W<w> H<h> F<num>:<den> Ip A0:0 C<colorspace>\n
        let line = format!(
            "YUV4MPEG2 W{w} H{h} F{fn_}:{fd} Ip A0:0 C{cs}\n",
            w = self.width,
            h = self.height,
            fn_ = self.frame_rate.num,
            fd = self.frame_rate.den,
            cs = self.colorspace,
        );
        self.output.write_all(line.as_bytes())?;
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("y4m muxer: write_header not called"));
        }
        if packet.data.len() != self.frame_size {
            return Err(Error::invalid(format!(
                "y4m muxer: frame size mismatch (got {} bytes, expected {})",
                packet.data.len(),
                self.frame_size
            )));
        }
        self.output.write_all(b"FRAME\n")?;
        self.output.write_all(&packet.data)?;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

// ───────────────────────── tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::NullCodecResolver;

    /// Build the smallest-possible Y4M file in memory: an 8×8 single-
    /// frame yuv420p stream.
    fn build_minimal_y4m_8x8(num_frames: usize) -> Vec<u8> {
        let header = b"YUV4MPEG2 W8 H8 F30:1 Ip A1:1 C420mpeg2\n";
        let frame_payload_len = 8 * 8 + 4 * 4 + 4 * 4;
        let mut out = Vec::new();
        out.extend_from_slice(header);
        for f in 0..num_frames {
            out.extend_from_slice(b"FRAME\n");
            for i in 0..frame_payload_len {
                out.push(((i + f * 13) & 0xff) as u8);
            }
        }
        out
    }

    #[test]
    fn probe_detects_magic() {
        let probe_data = ProbeData {
            buf: b"YUV4MPEG2 W16 H16 F25:1\n",
            ext: None,
        };
        assert_eq!(probe(&probe_data), 100);
        let probe_bad = ProbeData {
            buf: b"NOTAMAGIC ",
            ext: None,
        };
        assert_eq!(probe(&probe_bad), 0);
    }

    #[test]
    fn parse_header_roundtrip_basic() {
        let line = b"YUV4MPEG2 W64 H48 F30000:1001 Ip A1:1 C420jpeg XYSCSS=420JPEG";
        let hdr = parse_header(line).unwrap();
        assert_eq!(hdr.width, 64);
        assert_eq!(hdr.height, 48);
        assert_eq!(hdr.frame_rate, Rational::new(30000, 1001));
        assert_eq!(hdr.pixel_aspect, Rational::new(1, 1));
        assert_eq!(hdr.interlace, b'p');
        assert_eq!(hdr.colorspace, "420jpeg");
        assert_eq!(hdr.x_params.len(), 1);
        assert_eq!(
            hdr.x_params[0],
            ("XYSCSS".to_string(), "420JPEG".to_string())
        );
    }

    #[test]
    fn parse_header_rejects_missing_dimensions() {
        let no_w = b"YUV4MPEG2 H48 F25:1";
        assert!(parse_header(no_w).is_err());
        let no_h = b"YUV4MPEG2 W48 F25:1";
        assert!(parse_header(no_h).is_err());
    }

    #[test]
    fn frame_size_420_versus_422_versus_444() {
        let mut hdr = Y4mHeader {
            width: 16,
            height: 16,
            frame_rate: Rational::new(25, 1),
            pixel_aspect: Rational::new(1, 1),
            interlace: b'p',
            colorspace: "420jpeg".to_string(),
            x_params: Vec::new(),
        };
        assert_eq!(frame_size(&hdr).unwrap(), 16 * 16 + 8 * 8 + 8 * 8);
        hdr.colorspace = "422".to_string();
        assert_eq!(frame_size(&hdr).unwrap(), 16 * 16 + 8 * 16 + 8 * 16);
        hdr.colorspace = "444".to_string();
        assert_eq!(frame_size(&hdr).unwrap(), 16 * 16 * 3);
        hdr.colorspace = "mono".to_string();
        assert_eq!(frame_size(&hdr).unwrap(), 16 * 16);
        hdr.colorspace = "420p10".to_string();
        assert_eq!(frame_size(&hdr).unwrap(), (16 * 16 + 8 * 8 + 8 * 8) * 2);
    }

    #[test]
    fn demux_minimal_y4m_two_frames() {
        let bytes = build_minimal_y4m_8x8(2);
        let cur = std::io::Cursor::new(bytes);
        let mut dmx = open_demuxer(Box::new(cur), &NullCodecResolver).unwrap();
        assert_eq!(dmx.format_name(), "y4m");
        assert_eq!(dmx.streams().len(), 1);
        let s = &dmx.streams()[0];
        assert_eq!(s.params.codec_id, CodecId::new("rawvideo"));
        assert_eq!(s.params.width, Some(8));
        assert_eq!(s.params.height, Some(8));
        assert_eq!(s.params.pixel_format, Some(PixelFormat::Yuv420P));
        assert_eq!(s.params.frame_rate, Some(Rational::new(30, 1)));

        let p1 = dmx.next_packet().unwrap();
        assert_eq!(p1.pts, Some(0));
        assert_eq!(p1.data.len(), 8 * 8 + 4 * 4 + 4 * 4);
        let p2 = dmx.next_packet().unwrap();
        assert_eq!(p2.pts, Some(1));
        assert!(matches!(dmx.next_packet(), Err(Error::Eof)));
    }

    #[test]
    fn demux_preserves_x_params_in_metadata() {
        // Hand-built Y4M with an extension param.
        let header = b"YUV4MPEG2 W8 H8 F25:1 Ip A1:1 C420jpeg XCOLORRANGE=LIMITED\n";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(b"FRAME\n");
        bytes.resize(bytes.len() + 8 * 8 + 4 * 4 + 4 * 4, 0u8);
        let cur = std::io::Cursor::new(bytes);
        let dmx = open_demuxer(Box::new(cur), &NullCodecResolver).unwrap();
        let md = dmx.metadata();
        assert!(md.iter().any(|(k, v)| k == "XCOLORRANGE" && v == "LIMITED"));
        assert!(md.iter().any(|(k, v)| k == "colorspace" && v == "420jpeg"));
    }

    #[test]
    fn round_trip_8x8_yuv420p() {
        // Build 4 distinct frames of synthetic 8×8 yuv420p, mux to
        // memory, then demux and verify byte-exact equality.
        let frames: Vec<Vec<u8>> = (0..4)
            .map(|f| {
                let mut v = Vec::with_capacity(8 * 8 + 4 * 4 + 4 * 4);
                for y in 0..8 {
                    for x in 0..8 {
                        v.push((((x + y) * 7 + f * 11) & 0xff) as u8);
                    }
                }
                for u in 0..4 {
                    for x in 0..4 {
                        v.push(((u * 13 + x * 19 + f * 23) & 0xff) as u8);
                    }
                }
                for vv in 0..4 {
                    for x in 0..4 {
                        v.push(((vv * 17 + x * 29 + f * 5) & 0xff) as u8);
                    }
                }
                v
            })
            .collect();

        let mut params = CodecParameters::video(CodecId::new("rawvideo"));
        params.width = Some(8);
        params.height = Some(8);
        params.pixel_format = Some(PixelFormat::Yuv420P);
        params.frame_rate = Some(Rational::new(30, 1));
        let stream = StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, 30),
            duration: None,
            start_time: Some(0),
            params,
        };

        // Mux to a temp file, then read it back. The muxer demands a
        // `Box<dyn WriteSeek + 'static>`, which a borrowed in-memory
        // Cursor can't satisfy.
        let tmp = std::env::temp_dir().join("oxideav-basic-y4m-unit-420.y4m");
        let _ = std::fs::remove_file(&tmp);
        {
            let f = std::fs::File::create(&tmp).unwrap();
            let ws: Box<dyn WriteSeek> = Box::new(f);
            let mut mux = open_muxer(ws, std::slice::from_ref(&stream)).unwrap();
            mux.write_header().unwrap();
            for f in &frames {
                let pkt = Packet::new(0, stream.time_base, f.clone());
                mux.write_packet(&pkt).unwrap();
            }
            mux.write_trailer().unwrap();
        }

        let muxed = std::fs::read(&tmp).unwrap();
        // Expected on-the-wire shape. The muxer picks `420mpeg2` for
        // the generic limited-range Yuv420P pixel format.
        assert!(muxed.starts_with(b"YUV4MPEG2 W8 H8 F30:1 Ip A0:0 C420mpeg2\n"));

        // Demux back and compare frame-for-frame.
        let cur = std::io::Cursor::new(muxed);
        let mut dmx = open_demuxer(Box::new(cur), &NullCodecResolver).unwrap();
        for (i, want) in frames.iter().enumerate() {
            let p = dmx.next_packet().unwrap();
            assert_eq!(p.pts, Some(i as i64));
            assert_eq!(p.data, *want, "frame {i} payload mismatch");
        }
        assert!(matches!(dmx.next_packet(), Err(Error::Eof)));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn round_trip_16x16_yuv444p() {
        let mut frame = Vec::with_capacity(16 * 16 * 3);
        for plane in 0..3 {
            for y in 0..16 {
                for x in 0..16 {
                    frame.push(((plane * 31 + y * 7 + x * 11) & 0xff) as u8);
                }
            }
        }
        let mut params = CodecParameters::video(CodecId::new("rawvideo"));
        params.width = Some(16);
        params.height = Some(16);
        params.pixel_format = Some(PixelFormat::Yuv444P);
        params.frame_rate = Some(Rational::new(24, 1));
        let stream = StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, 24),
            duration: None,
            start_time: Some(0),
            params,
        };

        let tmp = std::env::temp_dir().join("oxideav-basic-y4m-unit-444.y4m");
        let _ = std::fs::remove_file(&tmp);
        {
            let f = std::fs::File::create(&tmp).unwrap();
            let ws: Box<dyn WriteSeek> = Box::new(f);
            let mut mux = open_muxer(ws, std::slice::from_ref(&stream)).unwrap();
            mux.write_header().unwrap();
            mux.write_packet(&Packet::new(0, stream.time_base, frame.clone()))
                .unwrap();
            mux.write_trailer().unwrap();
        }
        let muxed = std::fs::read(&tmp).unwrap();
        assert!(muxed.starts_with(b"YUV4MPEG2 W16 H16 F24:1 Ip A0:0 C444\n"));

        let cur = std::io::Cursor::new(muxed);
        let mut dmx = open_demuxer(Box::new(cur), &NullCodecResolver).unwrap();
        let p = dmx.next_packet().unwrap();
        assert_eq!(p.data, frame);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn muxer_rejects_wrong_frame_size() {
        let mut params = CodecParameters::video(CodecId::new("rawvideo"));
        params.width = Some(8);
        params.height = Some(8);
        params.pixel_format = Some(PixelFormat::Yuv420P);
        params.frame_rate = Some(Rational::new(25, 1));
        let stream = StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, 25),
            duration: None,
            start_time: Some(0),
            params,
        };
        let tmp = std::env::temp_dir().join("oxideav-basic-y4m-unit-bad.y4m");
        let _ = std::fs::remove_file(&tmp);
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = open_muxer(ws, std::slice::from_ref(&stream)).unwrap();
        mux.write_header().unwrap();
        let pkt = Packet::new(0, stream.time_base, vec![0u8; 100]);
        assert!(mux.write_packet(&pkt).is_err());
        let _ = std::fs::remove_file(&tmp);
    }
}
