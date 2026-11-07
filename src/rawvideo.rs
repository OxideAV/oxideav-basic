//! `rawvideo` codec: uncompressed pictures, one packet per frame.
//!
//! A packet holds the frame's image planes back to back, each plane's
//! rows tightly packed (no stride padding) — exactly the payload the
//! Y4M container stores after each `FRAME` marker and what other raw
//! carriers (`V_UNCOMPRESSED`, image-sequence sources) produce. The
//! plane geometry comes from the stream's `pixel_format` / `width` /
//! `height` ([`PixelFormat::plane_row_bytes`] /
//! [`PixelFormat::plane_dimensions`]).
//!
//! * **Decode**: split the packet into planes → [`VideoFrame`] with
//!   `stride == row bytes`.
//! * **Encode**: concatenate each image plane's visible rows (dropping
//!   any stride padding) → one packet.
//!
//! Palette formats (`Pal8`) are rejected: the palette is out-of-band
//! side data a raw payload has no room for.
//!
//! # Stream parameters and mid-stream changes
//!
//! A raw packet has no header, so the decoder's
//! [`Decoder::output_params`] is the stream's geometry (width, height,
//! pixel format, frame rate) from construction on and its generation
//! stays `0` — it never emits a [`Frame::StreamParams`] event.
//!
//! The encoder follows a [`Frame::StreamParams`] event that changes the
//! width, height, pixel format or frame rate: each packet is one
//! self-contained picture, so later packets carry the new geometry and
//! [`Encoder::output_params`] describes it from then on. An event for
//! a non-video stream, or for a geometry the codec cannot carry (empty
//! picture, palette format), is rejected with [`Error::Unsupported`]
//! and leaves the encoder unchanged.

use std::collections::VecDeque;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, Decoder, Encoder, Error,
    Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame, VideoPlane,
};

/// Codec id of uncompressed video.
pub const CODEC_ID: &str = "rawvideo";

/// Install the `rawvideo` decoder + encoder.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("rawvideo_sw")
        .with_lossless(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder),
    );
}

/// Validated picture geometry: per image plane `(row_bytes, rows)`.
#[derive(Debug, Clone)]
struct Layout {
    planes: Vec<(usize, usize)>,
    frame_bytes: usize,
}

impl Layout {
    fn from_params(params: &CodecParameters, who: &str) -> Result<Self> {
        let fmt = params
            .pixel_format
            .ok_or_else(|| Error::invalid(format!("rawvideo {who}: pixel_format is required")))?;
        let w = params
            .width
            .ok_or_else(|| Error::invalid(format!("rawvideo {who}: width is required")))?;
        let h = params
            .height
            .ok_or_else(|| Error::invalid(format!("rawvideo {who}: height is required")))?;
        if w == 0 || h == 0 {
            return Err(Error::invalid(format!(
                "rawvideo {who}: empty picture {w}x{h}"
            )));
        }
        if fmt == PixelFormat::Pal8 {
            return Err(Error::unsupported(format!(
                "rawvideo {who}: palette formats carry their palette out of band"
            )));
        }
        let mut planes = Vec::with_capacity(fmt.plane_count());
        let mut frame_bytes = 0usize;
        for p in 0..fmt.plane_count() {
            let row = fmt.plane_row_bytes(p, w);
            let dims = fmt.plane_dimensions(p, w, h);
            let (Some(row), Some((_, rows))) = (row, dims) else {
                return Err(Error::unsupported(format!(
                    "rawvideo {who}: no plane geometry for {fmt:?}"
                )));
            };
            let rows = rows as usize;
            frame_bytes = row
                .checked_mul(rows)
                .and_then(|n| frame_bytes.checked_add(n))
                .ok_or_else(|| Error::invalid(format!("rawvideo {who}: frame size overflow")))?;
            planes.push((row, rows));
        }
        Ok(Self {
            planes,
            frame_bytes,
        })
    }
}

/// Build a `rawvideo` decoder for `params`.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let layout = Layout::from_params(params, "decoder")?;
    Ok(Box::new(RawVideoDecoder {
        output: output_params_for(params),
        layout,
        pending: VecDeque::new(),
        eof: false,
    }))
}

/// The `rawvideo` stream description of `params`' picture geometry.
fn output_params_for(params: &CodecParameters) -> CodecParameters {
    let mut output = CodecParameters::video(CodecId::new(CODEC_ID));
    output.width = params.width;
    output.height = params.height;
    output.pixel_format = params.pixel_format;
    output.frame_rate = params.frame_rate;
    output
}

/// Build a `rawvideo` encoder for `params`.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let layout = Layout::from_params(params, "encoder")?;
    let output = output_params_for(params);
    Ok(Box::new(RawVideoEncoder {
        layout,
        output,
        queue: VecDeque::new(),
    }))
}

struct RawVideoDecoder {
    output: CodecParameters,
    layout: Layout,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Decoder for RawVideoDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.output.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.len() != self.layout.frame_bytes {
            return Err(Error::invalid(format!(
                "rawvideo: packet holds {} bytes, a frame is {}",
                packet.data.len(),
                self.layout.frame_bytes
            )));
        }
        self.pending.push_back(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.pop_front() else {
            return Err(if self.eof {
                Error::Eof
            } else {
                Error::NeedMore
            });
        };
        let mut planes = Vec::with_capacity(self.layout.planes.len());
        let mut off = 0usize;
        for &(row, rows) in &self.layout.planes {
            let len = row * rows;
            planes.push(VideoPlane {
                stride: row,
                data: pkt.data[off..off + len].to_vec(),
            });
            off += len;
        }
        Ok(Frame::Video(VideoFrame {
            pts: pkt.pts,
            planes,
        }))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // A seek resumes decoding: drop queued pictures and leave the
        // end-of-stream state, so an empty poll answers NeedMore.
        self.pending.clear();
        self.eof = false;
        Ok(())
    }

    fn output_pixel_format(&self) -> Option<PixelFormat> {
        self.output.pixel_format
    }

    fn output_params(&self) -> Option<CodecParameters> {
        Some(self.output.clone())
    }
}

struct RawVideoEncoder {
    layout: Layout,
    output: CodecParameters,
    queue: VecDeque<Packet>,
}

impl RawVideoEncoder {
    /// Follow a mid-stream geometry change (see the module docs).
    fn reconfigure(&mut self, p: &CodecParameters) -> Result<()> {
        if p.media_type != oxideav_core::MediaType::Video {
            return Err(Error::unsupported(
                "rawvideo encoder: stream-params event for a non-video stream",
            ));
        }
        let mut next = self.output.clone();
        next.width = p.width.or(next.width);
        next.height = p.height.or(next.height);
        next.pixel_format = p.pixel_format.or(next.pixel_format);
        next.frame_rate = p.frame_rate.or(next.frame_rate);
        let layout = Layout::from_params(&next, "encoder")
            .map_err(|e| Error::unsupported(format!("rawvideo encoder cannot follow: {e}")))?;
        self.layout = layout;
        self.output = next;
        Ok(())
    }
}

impl Encoder for RawVideoEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            Frame::StreamParams(p) => return self.reconfigure(p),
            _ => return Err(Error::invalid("rawvideo encoder requires video frames")),
        };
        let image = v.image_planes();
        if image.len() != self.layout.planes.len() {
            return Err(Error::invalid(format!(
                "rawvideo encoder: frame has {} planes, the format has {}",
                image.len(),
                self.layout.planes.len()
            )));
        }
        let mut data = Vec::with_capacity(self.layout.frame_bytes);
        for (plane, &(row, rows)) in image.iter().zip(&self.layout.planes) {
            let needed = plane
                .stride
                .checked_mul(rows - 1)
                .and_then(|n| n.checked_add(row));
            if plane.stride < row || needed.is_none_or(|n| plane.data.len() < n) {
                return Err(Error::invalid(
                    "rawvideo encoder: plane smaller than the picture geometry",
                ));
            }
            for r in 0..rows {
                let start = r * plane.stride;
                data.extend_from_slice(&plane.data[start..start + row]);
            }
        }
        // The frame pts is carried through unchanged; it counts in the
        // stream's time base, which for a constant-rate raw stream is
        // the frame period.
        let tb = self
            .output
            .frame_rate
            .filter(|r| r.num > 0 && r.den > 0)
            .map_or(TimeBase::new(1, 25), |r| TimeBase::new(r.den, r.num));
        let mut pkt = Packet::new(0, tb, data);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn params(fmt: PixelFormat, w: u32, h: u32) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new(CODEC_ID));
        p.pixel_format = Some(fmt);
        p.width = Some(w);
        p.height = Some(h);
        p
    }

    #[test]
    fn decode_splits_yuv420p_planes() {
        let mut d = make_decoder(&params(PixelFormat::Yuv420P, 5, 3)).unwrap();
        // 5x3 luma + 3x2 per chroma plane.
        let data: Vec<u8> = (0..(15 + 6 + 6)).map(|i| i as u8).collect();
        let mut pkt = Packet::new(0, TimeBase::new(1, 25), data);
        pkt.pts = Some(4);
        d.send_packet(&pkt).unwrap();
        let Frame::Video(v) = d.receive_frame().unwrap() else {
            panic!("video frame expected");
        };
        assert_eq!(v.pts, Some(4));
        assert_eq!(v.planes.len(), 3);
        assert_eq!((v.planes[0].stride, v.planes[0].data.len()), (5, 15));
        assert_eq!((v.planes[1].stride, v.planes[1].data.len()), (3, 6));
        assert_eq!(v.planes[2].data[0], 21);
        assert!(matches!(d.receive_frame(), Err(Error::NeedMore)));
        d.flush().unwrap();
        assert!(matches!(d.receive_frame(), Err(Error::Eof)));
    }

    #[test]
    fn decode_rejects_wrong_packet_size() {
        let mut d = make_decoder(&params(PixelFormat::Rgb24, 2, 2)).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), vec![0; 11]);
        assert!(d.send_packet(&pkt).is_err());
    }

    #[test]
    fn encode_drops_stride_padding_and_round_trips() {
        let p = params(PixelFormat::Gray8, 3, 2);
        let mut e = make_encoder(&p).unwrap();
        let frame = VideoFrame {
            pts: Some(9),
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![1, 2, 3, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0],
            }],
        };
        e.send_frame(&Frame::Video(frame)).unwrap();
        let pkt = e.receive_packet().unwrap();
        assert_eq!(pkt.data, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(pkt.pts, Some(9));
        let mut d = make_decoder(&p).unwrap();
        d.send_packet(&pkt).unwrap();
        let Frame::Video(v) = d.receive_frame().unwrap() else {
            panic!()
        };
        assert_eq!(v.planes[0].data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn decoder_states_geometry_and_resets_out_of_eof() {
        let mut d = make_decoder(&params(PixelFormat::Rgb24, 2, 1)).unwrap();
        let out = d.output_params().unwrap();
        assert_eq!((out.width, out.height), (Some(2), Some(1)));
        assert_eq!(out.pixel_format, Some(PixelFormat::Rgb24));
        assert_eq!(d.output_pixel_format(), Some(PixelFormat::Rgb24));
        assert_eq!(d.output_params_generation(), 0);
        d.flush().unwrap();
        assert!(matches!(d.receive_frame(), Err(Error::Eof)));
        d.reset().unwrap();
        assert!(matches!(d.receive_frame(), Err(Error::NeedMore)));
    }

    #[test]
    fn encoder_follows_geometry_change() {
        let mut e = make_encoder(&params(PixelFormat::Gray8, 2, 2)).unwrap();
        let ev = params(PixelFormat::Rgb24, 1, 1);
        e.send_frame(&Frame::StreamParams(Box::new(ev))).unwrap();
        assert_eq!(e.output_params().pixel_format, Some(PixelFormat::Rgb24));
        assert_eq!(e.output_params().width, Some(1));
        let frame = VideoFrame {
            pts: Some(0),
            planes: vec![VideoPlane {
                stride: 3,
                data: vec![7, 8, 9],
            }],
        };
        e.send_frame(&Frame::Video(frame)).unwrap();
        assert_eq!(e.receive_packet().unwrap().data, vec![7, 8, 9]);
    }

    #[test]
    fn encoder_refuses_unfollowable_events_unchanged() {
        let mut e = make_encoder(&params(PixelFormat::Gray8, 2, 2)).unwrap();
        for ev in [
            params(PixelFormat::Pal8, 2, 2),
            params(PixelFormat::Gray8, 0, 2),
            CodecParameters::audio(CodecId::new("pcm_s16le")),
        ] {
            let r = e.send_frame(&Frame::StreamParams(Box::new(ev)));
            assert!(matches!(r, Err(Error::Unsupported(_))), "{r:?}");
        }
        assert_eq!(e.output_params().pixel_format, Some(PixelFormat::Gray8));
        assert_eq!(e.output_params().width, Some(2));
    }

    #[test]
    fn encoder_rejects_overflowing_stride() {
        let mut e = make_encoder(&params(PixelFormat::Gray8, 2, 3)).unwrap();
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: usize::MAX,
                data: vec![0; 6],
            }],
        };
        assert!(e.send_frame(&Frame::Video(frame)).is_err());
    }

    #[test]
    fn rejects_palette_and_missing_geometry() {
        assert!(make_decoder(&params(PixelFormat::Pal8, 2, 2)).is_err());
        let mut p = params(PixelFormat::Yuv420P, 2, 2);
        p.pixel_format = None;
        assert!(make_encoder(&p).is_err());
    }
}
