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
  retained for compatibility). Also recognises the extended `INFO`
  sub-ID namespace catalogued in ExifTool's RIFF Info Tags table
  (`docs/container/riff/metadata/exiftool-riff-tags.html`): the
  per-stream audio-language slots `IAS1`..`IAS9` →
  `first_language`..`ninth_language`, the Windows-Media "more info"
  set (`IBSU` → `base_url`, `ICAS` → `default_audio_stream`, `ILGU` →
  `logo_url`, `ILIU` → `logo_icon_url`, `IMBI` →
  `more_info_banner_image`, `IMBU` → `more_info_banner_url`, `IMIT` →
  `more_info_text`, `IMIU` → `more_info_url`, `IWMU` →
  `watermark_url`), and the common production-credit / cataloguing
  tags (`ICDS` → `costume_designer`, `ICNM` → `cinematographer`,
  `ICNT` → `country`, `IDIT` → `date_time_original`, `IDST` →
  `distributed_by`, `IEDT` → `edited_by`, `IENC` → `encoded_by`,
  `ILNG` → `language`, `IMUS` → `music_by`, `IPDS` →
  `production_designer`, `IPRO` → `produced_by`, `IRIP` → `ripped_by`,
  `IRTD` → `rating`, `ISGN` → `secondary_genre`, `ISMP` → `time_code`,
  `ISTD` → `production_studio`, `ISTR` → `starring`, `IWRI` →
  `written_by`); unknown sub-IDs are skipped silently).
  Dispatches `WAVE_FORMAT_ALAW (0x0006)` /
  `WAVE_FORMAT_MULAW (0x0007)` to the `pcm_alaw` / `pcm_mulaw` codecs
  (host runtime applies G.711 decode). `WAVE_FORMAT_EXTENSIBLE (0xFFFE)`
  is parsed end-to-end — the 22-byte extension's `wValidBitsPerSample`,
  `dwChannelMask` and SubFormat GUID are surfaced through both
  `wav:fmt.*` metadata keys and typed accessors on the concrete
  `WavDemuxer`. The `dwChannelMask` bitmap is also decoded into a
  human-readable `SPEAKER_*` layout (`wav:fmt.channel_layout` +
  `WavDemuxer::channel_layout`), `+`-joined least-significant-bit-first
  per the 18 documented flag bits (`FRONT_LEFT 0x1` ..
  `TOP_BACK_RIGHT 0x20000`) in
  `docs/container/riff/waveformatextensible/ms-waveformatextensible.html`;
  bits above the highest defined flag are preserved as
  `UNKNOWN(0x...)`. Any SubFormat GUID built from the `KSMedia.h`
  `DEFINE_WAVEFORMATEX_GUID(x)` template — `{0000xxxx-0000-0010-8000-
  00AA00389B71}`, where the leading 16 bits carry the legacy
  `wFormatTag` `x` — resolves through the SAME `wFormatTag` dispatch the
  legacy `WAVEFORMATEX` path uses, implementing the documented
  `IS_VALID_WAVEFORMATEX_GUID` / `EXTRACT_WAVEFORMATEX_ID` macros from
  `docs/container/riff/waveformatextensible/ms-converting-format-tags-and-subformat-guids.md`.
  This generalises the four hand-listed `KSDATAFORMAT_SUBTYPE_*` GUIDs
  (PCM `0x0001`, IEEE_FLOAT `0x0003`, ALAW `0x0006`, MULAW `0x0007`) to
  every tag-derived GUID, and surfaces the embedded tag as
  `wav:fmt.subformat_tag` (e.g. an EXTENSIBLE file whose SubFormat is
  `{00000055-...}` is observably MP3-tagged even though this crate
  doesn't decode MP3). Template GUIDs whose embedded tag isn't a format
  this crate maps directly, and non-template (unknown) GUIDs, both
  synthesise a `wav:guid_<canonical-text>` id.
  `WavMuxOptions::with_extensible(mask)`
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
  `wav:adtl.ltxt.<dwName>.length` / `.purpose` (FOURCC) / `.text` plus
  its four §3 locale WORDs `.country` / `.language` / `.dialect` /
  `.code_page` (raw decimals, always emitted) with `.country_name` /
  `.language_name` resolved through the same §3 Chapter-2 tables the
  `CSET` chunk uses (emitted only when the code is in the enumerated
  set); the `file` (embedded media file) sub-chunk surfaces
  `wav:adtl.file.<dwName>.med_type` (FOURCC when printable, the
  spec-allowed zero as `0`, hex otherwise) and `.body_len` (embedded
  payload length — the `fileData` bytes themselves are not exposed
  through the string-typed metadata API); sub-chunks shorter than
  their fixed headers are skipped as opaque. The
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
  NUL/whitespace surface only `wav:ixml.body_len`. The `<axml>` chunk
  (EBU Tech 3285 Supplement 5) carries a UTF-8 XML document —
  typically an EBUCore wrapper around an `<audioFormatExtended>` ADM
  document or an ISRC identifier declaration — and surfaces through
  `wav:axml` (text payload trimmed at the first NUL + surrounding
  whitespace, schema-agnostic) and `wav:axml.body_len` (always
  emitted when the chunk is present, so a NUL-padded ADM reservation
  block reserved for in-place editing is observable). The `<bxml>`
  chunk (ITU-R BS.2088-2 §6) is the compressed-XML counterpart of
  `<axml>`: a 2-byte LE `fmtType` header (`0x0000` = uncompressed,
  `0x0001` = gzip per RFC 1952) precedes the (optionally compressed)
  XML payload. It surfaces `wav:bxml.fmt_type` (raw `0x%04X`),
  `wav:bxml.compression` (`none` / `gzip` label, omitted for
  private/future codes so the raw `fmt_type` stays authoritative),
  `wav:bxml.body_len` (full on-wire span including the 2-byte header,
  so a NUL-reserved in-place-edit block is observable), and — only for
  the uncompressed form — the `wav:bxml` text payload (trimmed at the
  first NUL + surrounding whitespace, exactly like `<axml>`).
  Compressed payloads are not inflated at the container layer (RFC 1952
  decode is left to a higher-level ADM-aware consumer); bodies shorter
  than the 2-byte `fmtType` header are skipped-as-opaque (only
  `body_len`). The `_PMX`
  chunk (Adobe XMP packet, the WAV/AVI carrier for an XMP serialised
  packet — FOURCC is little-endian "XMP_" reversed; catalogued in
  exiftool-riff-tags.html § "RIFF Main tags" entry `'_PMX'`, scope
  "AVI and WAV files") surfaces through `wav:xmp` (UTF-8 XMP packet
  text trimmed at the first NUL + surrounding whitespace, so writers
  that NUL-pad a fixed-size XMP region for in-place editing do not
  leak padding into the text key) and `wav:xmp.body_len` (always
  emitted when the chunk is present, so an XMP-aware reserved block
  is observable). Schema-agnostic — `<?xpacket begin=...?>` /
  `<?xpacket end=...?>` and the inner `x:xmpmeta` / RDF tree pass
  through unchanged. The `CSET`
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
  to the 64-bit `ds64.sampleCount` when it carries the sentinel. The
  `JUNK` (Filler) chunk (RIFF MCI §2 "JUNK (Filler) Chunk") is
  recognised end-to-end — the chunk body is defined as "no relevant
  data" so its bytes are not surfaced, but the demuxer accounts for
  every `JUNK` chunk seen: `wav:junk.count` (total number of `JUNK`
  chunks), `wav:junk.total_bytes` (cumulative payload bytes across all
  `JUNK` chunks; excludes the 8-byte chunk header and the word-align
  pad), and per-chunk `wav:junk.<n>.body_len` indexed zero-based by
  encounter order. Lets a downstream tool observe how much filler a
  writer reserved for in-place edits without pretending the bytes
  carry meaning. Multiple `JUNK` chunks are allowed; empty `JUNK`
  chunks (size = 0) still increment the count. Files with no `JUNK`
  chunk emit no `wav:junk.*` keys at all (absence is observable). The
  `slnt` (Silence) chunk (RIFF MCI §3 "Wave Data" —
  `slnt( dwSamples:DWORD )`) is recognised end-to-end: each chunk's
  4-byte `dwSamples` count of silent samples surfaces as
  `wav:slnt.<n>.samples` (zero-based by encounter order), the rolling
  aggregates `wav:slnt.count` / `wav:slnt.total_samples` accumulate, and
  no real zero/baseline samples are synthesised into the decoded stream
  (the §3 note is explicit that the correct fill value is
  context-dependent, not necessarily zero). Bodies shorter than the
  4-byte field are counted but treated as opaque (no `samples` key);
  over-length bodies decode the leading DWORD and tolerate trailing
  forward-extension bytes. Files with no `slnt` chunk emit no
  `wav:slnt.*` keys at all. The `LIST 'wavl'` wave-list waveform
  container (RIFF MCI §3 "Storage of WAVE Data" —
  `<wave-data> -> { <data-ck> | <data-list> }`,
  `<wave-list> -> LIST('wavl' { <data-ck> | <silence-ck> }... )`) is
  parsed end-to-end: the segmented form interleaves runs of PCM
  (`data` sub-chunks) with `slnt` silence-count markers, so the
  demuxer resolves the FIRST embedded `data` sub-chunk as the decode
  anchor (a `wavl`-form file is now decodable rather than yielding no
  audio) and surfaces every segment as `wav:wavl.segment_count` /
  `wav:wavl.data_count` / `wav:wavl.data_bytes` plus per-segment
  `wav:wavl.<n>.kind` (`data`/`slnt`) / `.length`. Embedded `slnt`
  segments feed the same `wav:slnt.*` accounting as top-level `slnt`
  chunks (silence is a count, never synthesised baseline samples);
  the `fact` chunk (required by §3 for `wavl`-form data) remains the
  authoritative duration. A silence-only `wavl` (no `data` segment)
  is rejected as having no waveform; odd-length `data` segments
  respect RIFF word-alignment. The Acidizer `acid` chunk (layout per the
  staged byte-indexed Acidizer table in
  `docs/container/riff/metadata/exiftool-riff-tags.html`) is supported
  on BOTH the read and write sides through the typed `wav::AcidChunk`
  struct (24-byte LE body: flags bit-field @0, root note @4, six
  reserved bytes carried verbatim @6..12, beats @12, meter @16, tempo
  @20, with `parse` / `to_bytes` and the five documented flag-bit
  helpers plus the 48..=71 root-note name table). The demuxer surfaces
  `wav:acid.*` keys (flags hex, each flag bit, root note + name,
  beats, meter, tempo, plus reserved-hex / body_len observability
  keys) and the typed view via `WavDemuxer::acid()`; the muxer writes
  the chunk via `WavMuxOptions::with_acid`. Truncated bodies are
  skipped-as-opaque; the mux→demux round-trip is pinned byte-for-byte
  in tests. The concrete demuxer is now publicly constructible via
  `wav::open_wav_demuxer` so every typed accessor is reachable without
  downcasting. The BW64/ADM `chna` (channel-allocation) chunk is now
  supported on BOTH read and write sides per the staged binary layout
  in `docs/container/riff/metadata/bs2088-chna-chunk-layout.md`
  (ITU-R BS.2088-2 §8.1): the typed `wav::ChnaChunk` / `wav::AudioId`
  structs (4-byte `numTracks`+`numUIDs` pre-amble followed by `N`
  fixed 40-byte `audioID` records — `trackIndex` u16 @0, `UID`[12]
  `ATU_…` @2, `trackRef`[14] `AT_…`/`AC_…` @14, `packRef`[11] `AP_…`
  or 11 NULs @28, `pad` @39) carry `parse` / `to_bytes` and the
  over-provisioning rule (`N = (ckSize−4)/40 ≥ numUIDs`, spare records
  marked `trackIndex == 0` round-trip verbatim). The demuxer surfaces
  `wav:chna.num_tracks` / `.num_uids` / `.record_count` /
  `.defined_count` plus per-defined-record
  `wav:chna.<n>.{track_index,uid,track_ref,pack_ref}` (fixed-width
  char fields rendered up to the first NUL, omitted when entirely
  NUL) and `wav:chna.body_len` when trailing extension bytes ride
  along; the typed view is reachable via `WavDemuxer::chna()` and the
  muxer writes the chunk via `WavMuxOptions::with_chna`. Bodies
  shorter than the 4-byte pre-amble are skipped-as-opaque; the
  mux→demux round-trip and the BS.2088-2 §8.3.1 stereo worked example
  are pinned byte-for-byte in tests.
- **slin** container: Asterisk-style headerless `.sln*` / `.slin*` raw
  S16LE PCM (extension drives the sample rate).
- **Y4M (YUV4MPEG2)** container: rawvideo demuxer + muxer for `.y4m` files,
  supporting 4:2:0 / 4:2:2 / 4:4:4 / mono at 8/10/12-bit. Header `X<key>=<val>`
  extensions are surfaced verbatim through `Demuxer::metadata`.
- **Filter primitive**: typed scalar Reinhard 2002 simple global
  tone-mapping operator (`Ld = L / (1 + L)`) with its closed-form inverse
  (`L = Ld / (1 − Ld)`). Scene-luminance and display-luminance domains are
  separated by distinct `SceneLuminance` / `DisplayLuminance` wrapper
  types, so callers can't accidentally swap pre- and post-tone-map values;
  constructors reject invalid inputs (negative, NaN, non-finite, or
  display ≥ 1.0). The forward map is a published mathematical fact
  transcribed from `docs/image/filter/tone-mapping-operators.md` §2.2.

## Usage

```toml
[dependencies]
oxideav-basic = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).
