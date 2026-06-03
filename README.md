# oxideav-basic

Simple standard codecs and containers for oxideav (PCM, WAV, ...)

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a pure-Rust media transcoding and streaming stack. Codec, container, and filter crates are implemented from the spec (no C codec libraries linked or wrapped, no `*-sys` crates). Optional hardware-engine crates (`oxideav-videotoolbox` / `-audiotoolbox` / `-vaapi` / `-vdpau` / `-nvidia` / `-vulkan-video`) bridge to OS APIs via runtime `libloading`; pass `--no-hwaccel` (or omit the `hwaccel` feature) to opt out.

## What's included

- **PCM codecs**: `pcm_u8`, `pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_f32le`,
  `pcm_f64le`.
- **WAV** container: RIFF/WAVE (plus EBU Tech 3306 / ITU-R BS.2088
  `RF64` and `BW64` 64-bit-extended forms) demuxer + muxer with
  `fmt`, `data`, and
  the full Microsoft RIFF MCI §3 "INFO List Chunk" baseline (23 sub-IDs
  from the 1991 spec: `IARL` → `archival_location`, `IART` → `artist`,
  `ICMS` → `commissioned`, `ICMT` → `comment`, `ICOP` → `copyright`,
  `ICRD` → `date`, `ICRP` → `cropped`, `IDIM` → `dimensions`, `IDPI` →
  `dpi`, `IENG` → `engineer`, `IGNR` → `genre`, `IKEY` → `keywords`,
  `ILGT` → `lightness`, `IMED` → `medium`, `INAM` → `title`, `IPLT` →
  `palette_setting`, `IPRD` → `album`, `ISBJ` → `subject`, `ISFT` →
  `encoder`, `ISHP` → `sharpness`, `ISRC` → `source`, `ISRF` →
  `source_form`, `ITCH` → `technician`; non-baseline `ITRK` → `track`
  retained for compatibility; unknown sub-IDs are skipped silently).
  Dispatches `WAVE_FORMAT_ALAW (0x0006)` /
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
  forward compatibility. When the top-level magic is `RF64` or `BW64`
  (the latter signalling an ADM-carrying file per ITU-R BS.2088) the
  demuxer expects a mandatory `ds64` chunk immediately after `WAVE`
  per EBU Tech 3306 §3 and Annex A.2. The 28-byte fixed prefix carries
  the 64-bit `riffSize`, `dataSize` and `sampleCount` overrides plus
  a `tableLength` count for an optional array of
  `(chunkId, chunkSize64)` records describing other non-`data` chunks
  that exceed 4 GiB. The 32-bit on-wire size field on any chunk may
  be the `0xFFFFFFFF` sentinel — `data` is promoted via the dedicated
  `dataSize` field, other chunk-IDs via the table lookup. Surfaces
  `wav:rf64.magic` (`RF64`/`BW64`), `wav:rf64.riff_size`,
  `wav:rf64.data_size`, `wav:rf64.sample_count`,
  `wav:rf64.table.count` plus per-entry `wav:rf64.table.<i>.id` /
  `.size` and `wav:rf64.body_len`. A sentinel without a `ds64`
  override is rejected as malformed; a `ds64` body shorter than 28
  bytes is rejected. The 32-bit legacy `fact.dwFileSize` is promoted
  to the 64-bit `ds64.sampleCount` when it carries the sentinel.
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
