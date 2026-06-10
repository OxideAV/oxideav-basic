# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- WAV demuxer completes `LIST adtl` (Associated Data List) coverage per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "Associated
  Data Chunk" — all four spec-defined sub-chunks are now parsed. The
  `file` sub-chunk (§3 "Embedded File Information",
  `file( <dwName:DWORD> <dwMedType:DWORD> <fileData:BYTE>... )`) is no
  longer skipped: `wav:adtl.file.<dwName>.med_type` renders the media
  type as FOURCC text when printable, the spec-allowed zero value as
  plain `0` ("This field can contain a zero value"), and hex otherwise;
  `wav:adtl.file.<dwName>.body_len` carries the embedded `fileData`
  payload length (the payload bytes themselves are not exposed through
  the string-typed metadata API — `body_len` keeps the attachment
  observable without pretending the parser interprets the inner
  format). The `ltxt` sub-chunk's four previously-undecoded locale
  WORDs now surface: `wav:adtl.ltxt.<dwName>.country` / `.language` /
  `.dialect` / `.code_page` (raw decimals, always emitted), with
  `.country_name` / `.language_name` resolved through the same §3
  Chapter-2 "Country Codes" / "Language and Dialect Codes" tables the
  `CSET` parser uses (emitted only when the code is in the spec's
  enumerated set; zero resolves to the tables' explicit `None` rows).
  Sub-chunks shorter than their fixed headers (8 bytes for `file`,
  20 for `ltxt`) remain skipped-as-opaque. Three new lib tests cover
  the `file` FOURCC + zero-med_type + payload-length path, the
  truncated-`file` opaque path, and the zero-locale `ltxt` path; the
  existing `ltxt` test now drives non-zero locale fields
  (UK / UK English / code page 1252) end-to-end.

- WAV demuxer recognises the `slnt` (Silence) chunk per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "Wave Data"
  (`<silence-ck> ➝ slnt( <dwSamples:DWORD> )` — "Count of silent
  samples"). Per the §3 note the chunk records a *count of silent
  samples* rather than carrying any PCM payload, and the correct
  playback fill value is context-dependent ("not necessarily a repeated
  zero volume or baseline sample"), so the parser surfaces accounting
  metadata without synthesising real zero/baseline samples into the
  decoded stream. Each encounter emits `wav:slnt.<n>.samples`
  (zero-based per-chunk `dwSamples` count) and updates the rolling
  aggregates `wav:slnt.count` (total `slnt` chunks seen) and
  `wav:slnt.total_samples` (cumulative silent-sample count). The §3
  grammar `<wave-data> ➝ { <data-ck> | <data-list> }` lets the chunk
  appear at the top level as a sibling of `data` (as well as inside a
  `wavl` LIST); the demuxer accounts for every top-level occurrence. A
  body shorter than the 4-byte `dwSamples` field is counted but treated
  as opaque — it contributes `0` to the running total and omits its
  per-chunk `samples` key, mirroring the other fixed-struct parsers;
  a body longer than 4 bytes decodes the leading DWORD and tolerates
  trailing forward-extension bytes. An odd-length body forces the
  standard RIFF §2 word-align pad and the `data` chunk that follows is
  located correctly. Files with no `slnt` chunk emit no `wav:slnt.*`
  keys at all — absence is observable. Seven new lib tests cover the
  single-chunk path, the multi-chunk accumulation path, the
  zero-samples in-range path, the no-slnt absence guard, the
  under-length opaque-body path, the over-length leading-DWORD path,
  and the coexistence path with `JUNK` + `LIST INFO` interleaved.
- WAV demuxer parses the `_PMX` chunk (Adobe XMP packet, the WAV/AVI
  carrier for an XMP serialised packet, catalogued in
  `docs/container/riff/metadata/exiftool-riff-tags.html` § "RIFF Main
  tags" — entry `'_PMX'`, family `XMP`, scope "AVI and WAV files").
  The FOURCC is little-endian "XMP_" reversed, the convention RIFF
  uses for chunks whose payload originates in a little-endian
  DWORD-aligned authoring tool; the payload is the XMP packet text
  exactly as it would appear in an XMP sidecar (`x:xmpmeta` wrapped
  in `<?xpacket begin=...?>` / `<?xpacket end=...?>` processing
  instructions). Surface shape mirrors the sibling third-party XML
  parsers (`iXML` / `<axml>`) so the consumer keeps a single mental
  model: the UTF-8 XMP text surfaces verbatim under `wav:xmp`
  (trimmed at the first NUL and surrounding whitespace so writers
  that NUL-pad a fixed-size XMP region for in-place editing do not
  leak padding into the text key), and the raw on-wire chunk-body
  length always surfaces under `wav:xmp.body_len` when the chunk is
  present — even for empty / NUL-only / whitespace-only bodies — so
  downstream tooling can distinguish "no `_PMX` chunk" from "an
  `_PMX` chunk reserved for later XMP authoring". An odd-length body
  forces the standard RIFF 1-byte pad and the `data` chunk that
  follows is located correctly. The XMP schema (RDF, namespace
  prefixes, xpacket processing instructions) is not interpreted at
  this layer; a higher-level XMP-aware crate can apply
  schema-specific decoding without re-walking the RIFF tree. Tests
  cover a canonical Adobe-style packet, NUL-padded reservation
  bodies, an empty body (`body_len = 0`, no text key),
  whitespace-only bodies, odd-length pad rounding, and the absence
  guard (a file without `_PMX` emits no `wav:xmp.*` keys).

- WAV demuxer parses the `<axml>` chunk per
  `docs/container/riff/metadata/ebu-tech3285s5-ADM.pdf` §3 ("AXML
  chunk definition"). The chunk carries a UTF-8 XML document
  (typically an EBUCore wrapper around an `<audioFormatExtended>`
  ADM document — see §4.2 — or an ISRC identifier declaration —
  §4.1) and is supplement-5's vehicle for transporting ADM
  metadata inside BWF / RF64 / BW64 files. The parser surfaces
  the textual payload verbatim under `wav:axml` (trimmed at the
  first NUL and surrounding whitespace, so writers that NUL-pad
  to reserve room for in-place editing of the ADM document do
  not leak the padding into the text key) and always emits the
  raw on-wire chunk-body length under `wav:axml.body_len`
  whenever the chunk is present — even for empty / NUL-only /
  whitespace-only bodies — so downstream tooling can distinguish
  "no `<axml>` chunk" from "an `<axml>` chunk reserved for later
  ADM authoring". An odd-length body forces the standard RIFF
  1-byte pad and the `data` chunk that follows is located
  correctly. The XML schema is not interpreted at this layer —
  the parser is schema-agnostic so a higher-level ADM-aware
  crate (or a downstream tool walking `wav:axml`) can apply the
  EBUCore / BS.2076 decoding without re-walking the RIFF tree.
  Six new lib tests cover the canonical EBUCore-wrapped ADM
  document path, the ISRC identifier example from §4.1, the
  NUL-padded reservation path, the empty-body absence guard, the
  whitespace-only placeholder path, and the odd-length-body
  padding regression guard.
- WAV demuxer recognises the `JUNK` (Filler) chunk per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §2 "JUNK
  (Filler) Chunk" ("A JUNK chunk represents padding, filler or
  outdated information. It contains no relevant data; it is a space
  filler of arbitrary size."). The chunk body is deliberately not
  surfaced — its bytes are spec-defined as having no semantic content
  — but the demuxer accounts for every `JUNK` chunk seen so a
  downstream tool can observe how much filler the producer reserved
  (commonly for in-place editing) without re-walking the file. Each
  encounter emits `wav:junk.<n>.body_len` (zero-based, per-chunk
  payload size) and updates the rolling aggregates `wav:junk.count`
  (total `JUNK` chunks seen) and `wav:junk.total_bytes` (cumulative
  payload size; excludes the 8-byte chunk header and the implicit
  word-align pad byte). Multiple `JUNK` chunks are allowed; empty
  bodies (`size = 0`) still increment the count. Files with no `JUNK`
  chunk emit no `wav:junk.*` keys at all — absence is observable.
  Odd-length bodies forward the chunk-walk past the implicit RIFF §2
  word-align pad byte correctly. Six new lib tests cover the single-
  chunk path, the multi-chunk accumulation path, the zero-length
  in-range path, the no-JUNK absence guard, the odd-length-body
  padding regression guard, and the coexistence path with `LIST INFO`
  + `CSET` interleaved around the JUNK chunks.
- New `filter` module exposing a typed scalar form of the Reinhard 2002
  simple global tone-mapping operator (`Ld = L / (1 + L)`) per
  `docs/image/filter/tone-mapping-operators.md` §2.2. The forward map is
  surfaced as `SceneLuminance::to_display`; the closed-form inverse
  `L = Ld / (1 − Ld)` is surfaced as `DisplayLuminance::to_scene`. The
  two newtypes separate scene (pre-tone-map, non-negative finite) and
  display (post-tone-map, `[0, 1)`) luminance domains so callers can't
  swap them at a function boundary. Constructors reject invalid inputs
  (negative scene values, NaN, non-finite, `Ld ≥ 1.0`). Round-trip is
  bit-exact at the three §2.2 reference points (`L = 0 → 0`,
  `L = 1 → 0.5`, `L = 3 → 0.75`) and stays within a relative error of
  `1e-9` across an 11-point span from `1e-9` to `1e6` of scene
  luminance. Six in-crate unit tests plus three integration tests on
  the public surface cover constructor rejection, known curve values,
  the `Ld < 1` asymptote, and round-trip in both directions.
- WAV demuxer recognises the `RF64` and `BW64` top-level form magics
  (EBU Tech 3306 v1 §3 and ITU-R BS.2088 / EBU Tech 3306 v2) and
  parses the mandatory `ds64` chunk that follows. The 28-byte fixed
  prefix decodes the 64-bit `riffSize`, `dataSize` and `sampleCount`
  overrides plus a `tableLength` count of `(chunkId, chunkSize64)`
  records — the optional table of per-chunk-ID 64-bit size overrides
  for any non-`data` chunk that exceeds 4 GiB. When a chunk's 32-bit
  on-wire size carries the `0xFFFFFFFF` sentinel the demuxer promotes
  it to 64-bit via the dedicated `ds64.dataSize` (for `data`) or the
  `ds64.table` lookup (for any other FOURCC). The legacy
  `fact.dwFileSize` is likewise promoted to `ds64.sampleCount` when
  it carries the sentinel. Surfaced under `wav:rf64.magic`,
  `wav:rf64.riff_size`, `wav:rf64.data_size`, `wav:rf64.sample_count`,
  `wav:rf64.table.count`, per-entry `wav:rf64.table.<i>.id` /
  `.size`, and `wav:rf64.body_len`. Five new tests cover the RF64
  ds64-promoted PCM path, the BW64 magic accepted on the same path,
  the non-`data` table-lookup path through a `LIST INFO` whose size
  is the sentinel, a sentinel-without-`ds64` rejection, and a
  short-body `ds64` rejection.
- WAV demuxer now resolves the complete Microsoft RIFF MCI §3 "INFO
  List Chunk" baseline — all 23 sub-IDs registered by the 1991 spec
  per `docs/container/riff/metadata/microsoft-riffmci.pdf` pp. 2-14
  to 2-16. The eleven previously-handled sub-IDs (`INAM`, `IART`,
  `IPRD`, `ICMT`, `ICRD`, `IGNR`, `ICOP`, `IENG`, `ITCH`, `ISFT`,
  `ISBJ`) keep their conventional short-form key names; the twelve
  additions surface under spec-derived snake_case names: `IARL` →
  `archival_location`, `ICMS` → `commissioned`, `ICRP` → `cropped`,
  `IDIM` → `dimensions`, `IDPI` → `dpi`, `IKEY` → `keywords`,
  `ILGT` → `lightness`, `IMED` → `medium`, `IPLT` →
  `palette_setting`, `ISHP` → `sharpness`, `ISRC` → `source`,
  `ISRF` → `source_form`. The non-baseline `ITRK` → `track`
  extension is retained for compatibility with the tag-writer
  ecosystem. Unknown sub-IDs are skipped silently rather than
  synthesised into ad-hoc keys, matching the §3 forward-compatibility
  rule that "new chunks may be defined".
- WAV demuxer parses the `CSET` (Character Set) chunk per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3
  "CSET (Character Set) Chunk". The canonical 8-byte body carries four
  16-bit little-endian fields — `wCodePage`, `wCountryCode`,
  `wLanguageCode`, `wDialect` — declaring the code page, country,
  language and dialect that file elements (notably the `LIST INFO`
  ZSTR sub-chunks) are interpreted under. The parser surfaces raw
  values under `wav:cset.code_page` / `.country` / `.language` /
  `.dialect`, resolves the §3 "Country Codes" and "Language and
  Dialect Codes" enumerations to human-readable
  `wav:cset.country_name` / `wav:cset.language_name` keys, and always
  emits `wav:cset.body_len` so writers that extend the chunk past the
  canonical 8 bytes are still observable to downstream tooling.
  All-zero fields are honoured as the spec's "use defaults" form (ISO
  8859/1 / USA / US English). Bodies shorter than 8 bytes are treated
  as opaque (only `body_len` is emitted); bodies longer than 8 bytes
  tolerate the trailing region for forward compatibility. CSET coexists
  with `LIST INFO`: a single file may carry both, and the INFO sub-IDs
  still resolve through the existing standard-tag mapping.
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
