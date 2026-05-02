//! Cross-decode an ffmpeg-generated `.y4m` through `Y4mDemuxer`.
//!
//! Build a 4-frame 64×64 testsrc clip with ffmpeg's `lavfi` and verify
//! that our demuxer reads the expected stream params (width, height,
//! pixel format, frame count). Skips silently when ffmpeg isn't
//! installed — same convention as the codec crates' interop tests.

use oxideav_basic::register_containers;
use oxideav_core::{CodecId, ContainerRegistry, Error, NullCodecResolver, PixelFormat, ReadSeek};
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

#[test]
fn cross_decode_ffmpeg_testsrc_64x64_4frames() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available — skipping y4m interop test");
        return;
    }

    let tmp = std::env::temp_dir().join("oxideav-basic-y4m-interop.y4m");
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
            "testsrc=size=64x64:rate=25",
            "-frames:v",
            "4",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&tmp)
        .status()
        .expect("ffmpeg invocation failed");
    assert!(status.success(), "ffmpeg failed to produce y4m fixture");

    let mut reg = ContainerRegistry::new();
    register_containers(&mut reg);

    // Probe (with extension hint) should resolve to our y4m container.
    let mut f: Box<dyn ReadSeek> = Box::new(std::fs::File::open(&tmp).unwrap());
    let detected = reg.probe_input(&mut *f, Some("y4m")).unwrap();
    assert_eq!(detected, "y4m");

    // Open the demuxer via the registry and walk the frames.
    let f: Box<dyn ReadSeek> = Box::new(std::fs::File::open(&tmp).unwrap());
    let mut dmx = reg.open_demuxer("y4m", f, &NullCodecResolver).unwrap();
    let s = &dmx.streams()[0];
    assert_eq!(s.params.codec_id, CodecId::new("rawvideo"));
    assert_eq!(s.params.width, Some(64));
    assert_eq!(s.params.height, Some(64));
    // ffmpeg emits `C420mpeg2` for `yuv420p` here, which we map to
    // `PixelFormat::Yuv420P` (limited-range planar 4:2:0). Treat all
    // three 8-bit YUV 4:2:0 limited-range mappings as acceptable to
    // stay robust against a future ffmpeg colour-tag change.
    let pf = s.params.pixel_format.expect("pixel_format set");
    assert!(
        matches!(pf, PixelFormat::Yuv420P | PixelFormat::YuvJ420P),
        "unexpected pixel format from ffmpeg y4m: {:?}",
        pf,
    );

    // Drain four frames, expect EOF immediately after.
    let expected_payload = 64 * 64 + 32 * 32 + 32 * 32;
    for i in 0..4 {
        let p = dmx.next_packet().expect("frame decode");
        assert_eq!(p.pts, Some(i as i64));
        assert_eq!(
            p.data.len(),
            expected_payload,
            "ffmpeg-frame {i} payload size mismatch",
        );
    }
    assert!(matches!(dmx.next_packet(), Err(Error::Eof)));

    let _ = std::fs::remove_file(&tmp);
}
