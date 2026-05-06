# oxideav-basic

Simple standard codecs and containers for oxideav (PCM, WAV, ...)

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a pure-Rust media transcoding and streaming stack. Codec, container, and filter crates are implemented from the spec (no C codec libraries linked or wrapped, no `*-sys` crates). Optional hardware-engine crates (`oxideav-videotoolbox` / `-audiotoolbox` / `-vaapi` / `-vdpau` / `-nvidia` / `-vulkan-video`) bridge to OS APIs via runtime `libloading`; pass `--no-hwaccel` (or omit the `hwaccel` feature) to opt out.

## What's included

- **PCM codecs**: `pcm_u8`, `pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_f32le`,
  `pcm_f64le`.
- **WAV** container: RIFF/WAVE demuxer + muxer with `fmt`, `data`, and
  `LIST/INFO` metadata support.
- **slin** container: Asterisk-style headerless `.sln*` / `.slin*` raw
  S16LE PCM (extension drives the sample rate).
- **Y4M (YUV4MPEG2)** container: rawvideo demuxer + muxer for `.y4m` files,
  supporting 4:2:0 / 4:2:2 / 4:4:4 / mono at 8/10/12-bit. Header `X<key>=<val>`
  extensions are surfaced verbatim through `Demuxer::metadata`.

## Usage

```toml
[dependencies]
oxideav-basic = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).
