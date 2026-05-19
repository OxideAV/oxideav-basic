# oxideav-basic

Simple standard codecs and containers for oxideav (PCM, WAV, ...)

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a pure-Rust media transcoding and streaming stack. Codec, container, and filter crates are implemented from the spec (no C codec libraries linked or wrapped, no `*-sys` crates). Optional hardware-engine crates (`oxideav-videotoolbox` / `-audiotoolbox` / `-vaapi` / `-vdpau` / `-nvidia` / `-vulkan-video`) bridge to OS APIs via runtime `libloading`; pass `--no-hwaccel` (or omit the `hwaccel` feature) to opt out.

## What's included

- **PCM codecs**: `pcm_u8`, `pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_f32le`,
  `pcm_f64le`.
- **WAV** container: RIFF/WAVE demuxer + muxer with `fmt`, `data`, and
  `LIST/INFO` metadata. Dispatches `WAVE_FORMAT_ALAW (0x0006)` /
  `WAVE_FORMAT_MULAW (0x0007)` to the `pcm_alaw` / `pcm_mulaw` codecs
  (host runtime applies G.711 decode). `WAVE_FORMAT_EXTENSIBLE (0xFFFE)`
  is parsed end-to-end — the 22-byte extension's `wValidBitsPerSample`,
  `dwChannelMask` and SubFormat GUID are surfaced through both
  `wav:fmt.*` metadata keys and typed accessors on the concrete
  `WavDemuxer`. Well-known `KSDATAFORMAT_SUBTYPE_*` GUIDs (PCM,
  IEEE_FLOAT, ALAW, MULAW) resolve to the same codec ids the legacy
  `WAVEFORMATEX` path would have produced; unknown GUIDs synthesise a
  `wav:guid_<canonical-text>` id. `WavMuxOptions::with_extensible(mask)`
  opts the muxer into writing a 40-byte EXTENSIBLE `fmt ` chunk.
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
