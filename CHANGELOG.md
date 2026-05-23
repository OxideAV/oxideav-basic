# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
