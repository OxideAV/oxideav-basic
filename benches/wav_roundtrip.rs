//! Dependency-free read/write micro-benchmark for the WAV hot path.
//!
//! `harness = false` → a plain `main` timing loop on stable, with no
//! external bench-harness crate. Times the two hot paths that dominate a
//! transcode front/back end:
//!
//! * **demux (read)** — walk the RIFF chunk table, then drain every PCM
//!   packet out of a ~4 MiB S16 stereo `data` chunk from an in-memory
//!   cursor;
//! * **mux (write)** — frame the same payload back out to an in-memory
//!   sink through the muxer.
//!
//! Run with `cargo bench` (or `cargo bench -p oxideav-basic`); it prints
//! MiB/s for each direction. Numbers are indicative, not a regression
//! gate — a plain wall-clock loop, so expect run-to-run variance.

use std::io::Cursor;
use std::time::Instant;

use oxideav_basic::pcm;
use oxideav_basic::wav::{open_muxer_with, open_wav_demuxer, WavMuxOptions};
use oxideav_core::{Demuxer, Error, Packet, SampleFormat, StreamInfo, TimeBase, WriteSeek};

fn build_stream() -> StreamInfo {
    let params = pcm::params(SampleFormat::S16, 2, 48_000).expect("valid S16 stereo params");
    StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 48_000),
        duration: None,
        start_time: Some(0),
        params,
    }
}

/// Frame `payload` into a complete WAV byte buffer, in memory. Setup only.
fn mux_to_vec(stream: &StreamInfo, payload: &[u8]) -> Vec<u8> {
    // A shared in-memory sink so the framed bytes can be recovered after
    // the boxed muxer (which requires a `'static` `WriteSeek`) drops.
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Clone)]
    struct SharedVec(Rc<RefCell<Cursor<Vec<u8>>>>);
    // The bench is single-threaded; the `Send` bound on `WriteSeek` is
    // satisfied by this wrapper's promise never to cross a thread.
    unsafe impl Send for SharedVec {}
    impl std::io::Write for SharedVec {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.borrow_mut().write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.0.borrow_mut().flush()
        }
    }
    impl std::io::Seek for SharedVec {
        fn seek(&mut self, from: std::io::SeekFrom) -> std::io::Result<u64> {
            self.0.borrow_mut().seek(from)
        }
    }

    let shared = SharedVec(Rc::new(RefCell::new(Cursor::new(Vec::new()))));
    {
        let ws: Box<dyn WriteSeek> = Box::new(shared.clone());
        let mut mux = open_muxer_with(ws, std::slice::from_ref(stream), WavMuxOptions::default())
            .expect("muxer opens");
        mux.write_header().expect("header");
        let pkt = Packet::new(0, stream.time_base, payload.to_vec());
        mux.write_packet(&pkt).expect("packet");
        mux.write_trailer().expect("trailer");
    }
    let inner = shared.0.borrow();
    inner.get_ref().clone()
}

fn main() {
    // ~4 MiB payload: 1,048,576 S16 stereo frames × 4 bytes.
    let frames: u32 = 1 << 20;
    let mut payload = Vec::with_capacity(frames as usize * 4);
    for i in 0..frames {
        let l = (i as i16).wrapping_mul(3);
        let r = (i as i16).wrapping_mul(7);
        payload.extend_from_slice(&l.to_le_bytes());
        payload.extend_from_slice(&r.to_le_bytes());
    }
    let stream = build_stream();
    let bytes = mux_to_vec(&stream, &payload);
    let mib = payload.len() as f64 / (1024.0 * 1024.0);
    let iters = 40;

    // ---- demux (read) hot path ----
    let mut sink = 0u64;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut dmx = open_wav_demuxer(Box::new(Cursor::new(bytes.clone()))).expect("demux opens");
        loop {
            match dmx.next_packet() {
                Ok(p) => sink = sink.wrapping_add(p.data.len() as u64),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
    }
    let read_thru = mib * iters as f64 / t0.elapsed().as_secs_f64();

    // ---- mux (write) hot path ----
    let mut wsink = 0u64;
    let t1 = Instant::now();
    for _ in 0..iters {
        let out = mux_to_vec(&stream, &payload);
        wsink = wsink.wrapping_add(out.len() as u64);
    }
    let write_thru = mib * iters as f64 / t1.elapsed().as_secs_f64();

    // Consume the accumulators so the loops aren't optimised away.
    println!("wav demux (read) : {read_thru:8.1} MiB/s   [checksum {sink}]");
    println!("wav mux   (write): {write_thru:8.1} MiB/s   [checksum {wsink}]");
}
