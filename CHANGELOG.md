# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- WAV demuxer parses the `iXML` chunk (the third-party
  production-recorder metadata block catalogued in
  `docs/container/riff/metadata/exiftool-riff-tags.html` § `iXML`
  and discussed in
  `docs/container/riff/metadata/README.md` § "iXML"). The chunk
  carries a UTF-8 XML document — `IXML_VERSION`, `PROJECT`,
  `SCENE`, `TAKE`, `TAPE`, `NOTE`, `UBITS`, `FILE_UID`, a `BWF`
  sub-group mirroring the `bext` fields and a `TRACK_LIST` of
  per-track `NAME` / `FUNCTION` / `CHANNEL_INDEX` mappings. The
  parser surfaces the text payload verbatim under `wav:ixml`
  (trimmed at the first NUL and surrounding whitespace so writers
  that NUL-pad the body to a fixed size for in-place editing do
  not surface spurious trailing bytes) and always emits the raw
  on-wire chunk-body length under `wav:ixml.body_len` whenever the
  chunk is present — even for empty / NUL-only / whitespace-only
  bodies — so downstream tooling can distinguish "no `iXML`
  chunk" from "an `iXML` chunk reserved for later population". An
  odd-length body forces the standard RIFF 1-byte pad and the
  `data` chunk that follows is located correctly.

## [0.0.8](https://github.com/OxideAV/oxideav-basic/compare/v0.0.7...v0.0.8) - 2026-05-30

### Other

- parse plst (Playlist) chunk per Microsoft RIFF MCI §3
- parse smpl (Sampler) + inst (Instrument) chunks
- parse cue + LIST adtl per Microsoft RIFF MCI §3
- parse bext Broadcast Audio Extension chunk (EBU Tech 3285)
- fix intra-doc links to WavMuxOptions / WavDemuxer methods
- A-law/μ-law tag 6/7 + WAVEFORMATEXTENSIBLE end-to-end

### Added

- WAV demuxer parses the `fact` chunk per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 ("FACT
  Chunk"). The 4-byte `dwFileSize` field (per-channel sample
  count) surfaces under `wav:fact.sample_count` as a decimal
  string and becomes the authoritative `StreamInfo::duration`
  for the open stream — important for non-PCM WAV streams where
  the `data_size / block_align` heuristic is meaningless because
  one byte of payload no longer maps to one sample. For PCM
  streams the heuristic and the `fact` value should agree; when
  they don't (compressed bitstream riding a non-PCM
  `wFormatTag`, or a writer that lied about the count) the
  parser additionally surfaces `wav:fact.mismatch` with both
  numbers so a downstream tool can flag the file rather than
  silently trusting one over the other. The spec explicitly
  reserves trailing bytes past `dwFileSize` for future extension
  fields, so a body longer than 4 bytes surfaces its total
  length under `wav:fact.body_len` (the extension bytes
  themselves are opaque to this parser by spec); a body shorter
  than 4 bytes is treated as opaque-and-skipped without
  panicking the parser. The WAV muxer now emits a `fact` chunk
  between `fmt ` and `data` for every non-PCM on-wire
  `wFormatTag` (G.711 A-law / μ-law and the EXTENSIBLE
  `0xFFFE` escape hatch) — RIFF MCI §3 says the chunk is
  required for any `wFormatTag != WAVE_FORMAT_PCM`; for plain
  PCM where the chunk is optional we skip emitting it to keep
  the post-r193 PCM muxer output byte-identical to pre-r193.
- WAV demuxer parses the `plst` (Playlist) chunk per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 ("Playlist
  Chunk"). The 4-byte `dwSegments` count is followed by N × 12-byte
  `<play-segment>` records (`dwName`, `dwLength`, `dwLoops`). Each
  segment surfaces under `wav:plst.count` plus per-segment
  `wav:plst.<n>.cue_id` / `.length` / `.loops` keys. Unlike the `cue `
  chunk which keys by `dwName`, the playlist is keyed by zero-based
  segment position because the spec explicitly allows a single cue
  id to appear in multiple segments (a cue replayed N times = N
  segments with identical `dwName`), so a `dwName`-indexed key would
  collide. A `dwSegments` count that exceeds the chunk body is
  clamped to the records that actually fit; bodies shorter than the
  4-byte segment-count header are treated as opaque and skipped.
- WAV demuxer parses the `smpl` (Sampler) and `inst` (Instrument)
  chunks per `docs/container/riff/metadata/exiftool-riff-tags.html` §
  "RIFF Sampler Tags" / "RIFF Instrument Tags" and summarised in
  `docs/container/riff/metadata/README.md` § "Sampler / Instrument
  chunks". `smpl` is a 36-byte fixed header (`Manufacturer`,
  `Product`, `SamplePeriod`, `MIDIUnityNote`, `MIDIPitchFraction`,
  `SMPTEFormat`, `SMPTEOffset`, `cSampleLoops`, `cbSamplerData`)
  followed by N × 24-byte loop records (`dwCuePointID`, `dwType`,
  `dwStart`, `dwEnd`, `dwFraction`, `dwPlayCount`). `inst` is a
  7-byte fixed struct (`UnshiftedNote`, `FineTune`, `Gain`, `LowNote`,
  `HighNote`, `LowVelocity`, `HighVelocity`). Both surface through
  `Demuxer::metadata` under `wav:smpl.*` / `wav:inst.*` keys; the
  `SMPTEOffset` `DWORD` is rendered as canonical `HH:MM:SS:FF`, and
  `FineTune` / `Gain` are decoded as signed `i8` (so `-3` cents shows
  as `-3`, not `253`). A `cSampleLoops` count exceeding what the chunk
  body actually carries is clamped to the records that fit; bodies
  shorter than the 36-byte (`smpl`) or 7-byte (`inst`) fixed header
  are treated as opaque and skipped.
- WAV demuxer parses the `cue ` chunk and the `LIST adtl` (Associated
  Data List) sub-chunks per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3
  ("Cue-Points Chunk" + "Playlist Chunk" + "Associated Data Chunk").
  The cue-point table surfaces under `wav:cue.count` plus per-point
  `wav:cue.<dwName>.position` / `.fcc_chunk` / `.chunk_start` /
  `.block_start` / `.sample_offset` (all values as decimal strings;
  `fcc_chunk` rendered as a 4-byte ASCII FOURCC when printable). The
  `LIST adtl` body's `labl` / `note` / `ltxt` sub-chunks surface
  under `wav:adtl.labl.<dwName>` / `wav:adtl.note.<dwName>` /
  `wav:adtl.ltxt.<dwName>.{length,purpose,text}`. `file` sub-chunks
  (embedded media files) are skipped — their bytes do not fit the
  string-typed metadata API. A `dwCuePoints` count that exceeds the
  chunk body is clamped to the records that actually fit so a writer
  that lies about the count cannot panic the parser; `labl` / `note`
  shorter than the 4-byte `dwName` header and `ltxt` shorter than the
  20-byte fixed header are treated as opaque and skipped.
- WAV demuxer parses the `bext` Broadcast Audio Extension chunk per
  `docs/container/riff/metadata/ebu-tech3285-bwf.pdf` (EBU Tech 3285 v2
  §2.3 `BROADCAST_EXT`). The 602-byte fixed struct plus variable
  `CodingHistory` is surfaced through `Demuxer::metadata` under
  `wav:bext.description` / `.originator` / `.originator_reference` /
  `.origination_date` / `.origination_time` / `.time_reference` (64-bit
  sample count reassembled from the low/high `DWORD`s) / `.version` /
  `.umid` (hex-encoded SMPTE-330M UMID, v1+ only when non-zero) /
  `.coding_history`. For BWF v2 the five loudness `WORD`s are decoded
  from the §2.4 `round(100 × value)` fixed-point to two decimals under
  `.loudness_value` / `.loudness_range` / `.max_true_peak_level` /
  `.max_momentary_loudness` / `.max_short_term_loudness`. Loudness and
  UMID keys are omitted for v0/v1 streams that leave those fields zero;
  a `bext` chunk shorter than 602 bytes is skipped as opaque.
- WAV demuxer dispatches `WAVE_FORMAT_ALAW (0x0006)` and
  `WAVE_FORMAT_MULAW (0x0007)` streams to the `pcm_alaw` / `pcm_mulaw`
  codecs (host runtime applies G.711 decode through `oxideav-g711`).
  Decoded `SampleFormat` hint surfaces as `S16`; `bit_rate` reflects
  the on-wire 8-bit rate (not the post-decode S16 rate).
- WAV `WAVE_FORMAT_EXTENSIBLE (0xFFFE)` end-to-end handling per
  `docs/container/riff/waveformatextensible/README.md`: the 22-byte
  extension's `wValidBitsPerSample`, `dwChannelMask` and SubFormat
  GUID are parsed and exposed both through typed accessors on
  `WavDemuxer` (`format_tag`, `valid_bits_per_sample`, `channel_mask`,
  `subformat`, `subformat_text`) and through `Demuxer::metadata`
  under the keys `wav:fmt.valid_bits_per_sample` /
  `wav:fmt.channel_mask` / `wav:fmt.subformat`. Well-known
  `KSDATAFORMAT_SUBTYPE_*` GUIDs (PCM, IEEE_FLOAT, ALAW, MULAW)
  resolve to the legacy codec ids; unknown GUIDs synthesise a
  `wav:guid_<canonical-text>` id so downstream `make_decoder`
  failures name the actual GUID rather than the opaque `0xFFFE` tag.
- WAV muxer `open_muxer_with` + `WavMuxOptions::with_extensible(mask)`
  emits a 40-byte `WAVEFORMATEXTENSIBLE` `fmt ` chunk with caller-
  supplied `dwChannelMask`. `wValidBitsPerSample` defaults to the
  container `wBitsPerSample` and SubFormat defaults to the well-known
  GUID for the codec; both can be overridden via
  `with_valid_bits_per_sample` / `with_subformat`.
- New public constants `WAVE_FORMAT_PCM` / `WAVE_FORMAT_IEEE_FLOAT` /
  `WAVE_FORMAT_ALAW` / `WAVE_FORMAT_MULAW` / `WAVE_FORMAT_EXTENSIBLE`
  in the `wav` module for muxer callers.

### Fixed

- WAV demuxer rejects `WAVE_FORMAT_EXTENSIBLE` streams whose
  `cbSize < 22` (the spec mandates a 22-byte extension; previously
  the demuxer silently dropped the extension fields).
- WAV demuxer stamps the on-wire `wFormatTag` onto
  `CodecParameters.tag` so consumers can distinguish legacy
  `WAVEFORMATEX` from EXTENSIBLE round-trips without reparsing.

## [0.0.7](https://github.com/OxideAV/oxideav-basic/compare/v0.0.6...v0.0.7) - 2026-05-06

### Other

- reframe FFI claim — HW-engine crates use OS FFI by necessity
- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-basic/pull/502))

## [0.0.6](https://github.com/OxideAV/oxideav-basic/compare/v0.0.5...v0.0.6) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- add YUV4MPEG2 demuxer + muxer
- drop unused PcmDecoder.sample_rate field (slim-frame leftover)
- adopt slim AudioFrame shape
- pin release-plz to patch-only bumps

### Added

- Y4M (YUV4MPEG2) raw-video demuxer + muxer (`y4m` module). Handles
  4:2:0 / 4:2:2 / 4:4:4 / mono at 8/10/12-bit, preserves header
  `X<key>=<val>` extensions in `Demuxer::metadata`, and probes on the
  `YUV4MPEG2 ` magic. Frames are emitted as `rawvideo` packets.

## [0.0.5](https://github.com/OxideAV/oxideav-basic/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.4](https://github.com/OxideAV/oxideav-basic/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- bump oxideav-container dep to "0.1"
- drop Cargo.lock — this crate is a library
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()
