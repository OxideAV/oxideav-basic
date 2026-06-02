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
  opts the muxer into writing a 40-byte EXTENSIBLE `fmt ` chunk. The
  `bext` Broadcast Audio Extension chunk (EBU Tech 3285) is parsed and
  surfaced through `wav:bext.*` metadata keys — description, originator,
  origination date/time, 64-bit `TimeReference`, BWF version, SMPTE-330M
  UMID (v1+) and the v2 loudness fields (`LoudnessValue`,
  `LoudnessRange`, `MaxTruePeakLevel`, `MaxMomentaryLoudness`,
  `MaxShortTermLoudness`, each ×100 fixed-point rendered to two
  decimals) plus `CodingHistory`. The `fact` chunk (RIFF MCI §3
  "FACT Chunk") is parsed — `dwFileSize` (per-channel sample count)
  surfaces as `wav:fact.sample_count` and becomes the authoritative
  `StreamInfo::duration` (matters for compressed streams where
  `data_size / block_align` is meaningless); future-extension bytes
  past the 4-byte fixed field surface their total under
  `wav:fact.body_len`; a fact-vs-heuristic mismatch surfaces as
  `wav:fact.mismatch`. The muxer emits a `fact` chunk for every
  non-PCM `wFormatTag` (G.711 A-law/μ-law and the EXTENSIBLE escape
  hatch) per spec, and skips it for plain PCM where it is optional.
  The `cue ` chunk, `plst` (Playlist)
  chunk and `LIST adtl` (Associated Data List) sub-chunks are parsed
  per Microsoft RIFF MCI §3 — cue points surface as `wav:cue.count`
  plus per-point `wav:cue.<dwName>.position` / `.fcc_chunk` /
  `.chunk_start` / `.block_start` / `.sample_offset`; playlist
  segments surface as `wav:plst.count` plus per-segment
  `wav:plst.<n>.cue_id` / `.length` / `.loops` (zero-based segment
  index `<n>` because a single cue id can be replayed by multiple
  playlist entries); `labl` / `note` text sub-chunks surface as
  `wav:adtl.labl.<dwName>` / `wav:adtl.note.<dwName>`; the `ltxt`
  (text-with-segment-length) sub-chunk surfaces as
  `wav:adtl.ltxt.<dwName>.length` / `.purpose` (FOURCC) / `.text`. The
  `smpl` (Sampler) and `inst` (Instrument) chunks surface through
  `wav:smpl.*` (manufacturer / product / sample_period / midi_unity_note
  / midi_pitch_fraction / smpte_format / smpte_offset rendered as
  `HH:MM:SS:FF` / sampler_data_len / num_sample_loops + per-loop
  `wav:smpl.loop.<n>.{cue_point_id,type,start,end,fraction,play_count}`)
  and `wav:inst.{unshifted_note,fine_tune,gain,low_note,high_note,
  low_velocity,high_velocity}` (signed `fine_tune` / `gain` decoded as
  `i8`). Loop counts that exceed the chunk body are clamped; bodies
  shorter than the 36-byte `smpl` / 7-byte `inst` fixed header are
  treated as opaque. The `iXML` third-party metadata block (the
  production-recorder schema catalogued in ExifTool's RIFF tag
  table) is surfaced through `wav:ixml` (UTF-8 text payload, trimmed
  at the first NUL + surrounding whitespace) and `wav:ixml.body_len`
  (raw on-wire chunk size, always emitted when the chunk is present
  so a NUL-padded "reserved for in-place editing" region is still
  visible to downstream tooling); bodies that are empty or entirely
  NUL/whitespace surface only `wav:ixml.body_len`. The `CSET`
  (Character Set) chunk (RIFF MCI §3 "CSET Chunk") is parsed end-to-end:
  `wCodePage` / `wCountryCode` / `wLanguageCode` / `wDialect` (each a
  16-bit LE field) surface under `wav:cset.code_page` / `.country` /
  `.language` / `.dialect`, the §3 country and `(language, dialect)`
  tables resolve to human-readable `wav:cset.country_name` /
  `wav:cset.language_name` keys, and `wav:cset.body_len` is always
  emitted (so writers that extend the chunk past its canonical 8-byte
  struct are observable). Bodies shorter than 8 bytes are treated as
  opaque; bodies longer than 8 bytes tolerate the trailing region for
  forward compatibility.
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
