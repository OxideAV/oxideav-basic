//! Minimal pure-Rust RIFF/WAVE container.
//!
//! Supports reading and writing linear PCM streams via the `pcm_*` codecs
//! (`pcm_u8`, `pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_f32le`,
//! `pcm_f64le`) and dispatches `WAVE_FORMAT_ALAW (0x0006)` /
//! `WAVE_FORMAT_MULAW (0x0007)` streams to the external `pcm_alaw` /
//! `pcm_mulaw` codecs (provided by `oxideav-g711`, applied at decode
//! time by the host runtime — this crate's WAV layer only handles
//! framing).
//!
//! ## WAVEFORMATEXTENSIBLE (`wFormatTag = 0xFFFE`)
//!
//! Per `docs/container/riff/waveformatextensible/README.md` the demuxer
//! parses the 22-byte extension and surfaces:
//!
//! - `wValidBitsPerSample` — actual bit precision (may differ from the
//!   `wBitsPerSample` container size for 24-in-32-bit PCM).
//! - `dwChannelMask` — `SPEAKER_*` bitmap describing the channel
//!   ordering of the interleaved PCM byte stream.
//! - `SubFormat` GUID — the codec identifier when the legacy
//!   `wFormatTag` is the EXTENSIBLE escape hatch.
//!
//! Well-known `KSDATAFORMAT_SUBTYPE_*` GUIDs (PCM, IEEE_FLOAT, ALAW,
//! MULAW) are mapped to the same codec ids the legacy
//! `WAVEFORMATEX` path would have produced. Unknown GUIDs synthesise a
//! `wav:guid_<canonical-text>` codec id so downstream
//! `CodecRegistry::make_decoder` lookups fail cleanly naming the
//! actual GUID.
//!
//! The extension fields are also exposed verbatim through
//! `Demuxer::metadata` under the keys
//! `wav:fmt.valid_bits_per_sample` / `wav:fmt.channel_mask` /
//! `wav:fmt.channel_layout` / `wav:fmt.subformat` (matching the
//! round-75 `oxideav-avi` shape, but single-stream so no per-stream
//! index). `wav:fmt.channel_layout` is the `dwChannelMask` bitmap
//! decoded into a `+`-separated list of `SPEAKER_*` positions per
//! `docs/container/riff/waveformatextensible/ms-waveformatextensible.html`.

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, Error, MediaType, Packet, Result, SampleFormat,
    StreamInfo, TimeBase,
};
use oxideav_core::{ContainerRegistry, Demuxer, Muxer, ReadSeek, WriteSeek};
use std::io::{Read, Seek, SeekFrom, Write};

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("wav", open_demuxer);
    reg.register_muxer("wav", open_muxer);
    reg.register_extension("wav", "wav");
    reg.register_extension("wave", "wav");
    reg.register_probe("wav", probe);
}

/// `RIFF....WAVE` (standard), `RF64....WAVE` (EBU Tech 3306 §3
/// 64-bit-extended) or `BW64....WAVE` (ITU-R BS.2088 ADM-carrying
/// 64-bit-extended) — all unambiguous when present.
fn probe(p: &oxideav_core::ProbeData) -> u8 {
    if p.buf.len() < 12 {
        return 0;
    }
    let magic = &p.buf[0..4];
    let is_known_form = magic == b"RIFF" || magic == b"RF64" || magic == b"BW64";
    if is_known_form && &p.buf[8..12] == b"WAVE" {
        100
    } else {
        0
    }
}

/// The 32-bit sentinel value placed in the legacy `RIFF`/`data`/`fact`
/// size fields of an RF64 or BW64 file to signal "don't use this 32-bit
/// value, look up the real 64-bit value in `ds64`" (EBU Tech 3306 §3
/// and Annex A.2 `RF64Chunk` / `DataSize64Chunk` comments).
const SIZE64_SENTINEL: u32 = 0xFFFF_FFFF;

/// The body length of the fixed (table-less) `ds64` chunk: `riffSize`
/// (8) + `dataSize` (8) + `sampleCount`/`dummy` (8) + `tableLength`
/// (4) = 28 bytes (ITU-R BS.2088-2 §4.1 `DataSize64Chunk`; EBU Tech
/// 3306 v2 Annex A.2). The write side reserves exactly this much so a
/// `JUNK` placeholder can be promoted to a `ds64` chunk in place
/// (BS.2088-2 §3.6 "File structure with JUNK chunk" + §4.3).
const DS64_FIXED_BODY_LEN: u32 = 28;

/// `ds64` chunk decoded body. Populated only when the top-level magic
/// is `RF64` or `BW64`; for plain `RIFF`/WAVE the file uses 32-bit
/// sizes throughout and `Ds64` is unused.
///
/// Layout per EBU Tech 3306 v1 Annex A.2 `DataSize64Chunk`:
///
/// ```text
/// ds64
///   riffSize:    u64   // replaces the 32-bit RIFF size when sentinel
///   dataSize:    u64   // replaces the 32-bit `data` chunk size
///   sampleCount: u64   // replaces the 32-bit `fact` sample count
///   tableLength: u32   // number of `ChunkSize64` entries that follow
///   table:       [ChunkSize64; tableLength]
/// ```
///
/// Each `ChunkSize64` is `{ chunkId: [u8; 4], chunkSize: u64 }` and
/// gives a 64-bit size override for one non-`data` chunk-ID present
/// elsewhere in the file (the spec calls out that an LEVL chunk over
/// 4 GiB is the typical realistic case at ~512 GiB of audio payload).
#[derive(Default)]
struct Ds64 {
    /// 64-bit promotion of the top-level form-magic size field.
    /// Surfaced through metadata for downstream introspection; the
    /// demuxer doesn't use it directly because chunk-walking is
    /// driven by the per-chunk `data` size and individual chunk
    /// sizes (with sentinel resolution against `table`).
    #[allow(dead_code)]
    riff_size: u64,
    data_size: u64,
    sample_count: u64,
    table: Vec<([u8; 4], u64)>,
}

/// Parse a `ds64` chunk body. Surfaces every decoded field through
/// `metadata` for round-tripping. The `data` and `fact` sentinel
/// promotion is applied by the caller (which already holds the
/// 32-bit-field values from the regular chunk walk).
///
/// Per EBU Tech 3306 v1 Annex A.2: the fixed 28-byte prefix is
/// mandatory; the `table` array may be empty. Bodies shorter than 28
/// bytes are rejected as malformed; bodies longer than `28 + 12 *
/// tableLength` retain the trailing region for forward compatibility
/// (the body_len key surfaces it).
fn parse_ds64_chunk(buf: &[u8], out: &mut Vec<(String, String)>) -> Result<Ds64> {
    if buf.len() < 28 {
        return Err(Error::invalid("RF64 ds64 chunk shorter than 28 bytes"));
    }
    let riff_size = u64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]);
    let data_size = u64::from_le_bytes([
        buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
    ]);
    let sample_count = u64::from_le_bytes([
        buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
    ]);
    let table_len = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]) as usize;

    out.push(("wav:rf64.riff_size".to_string(), riff_size.to_string()));
    out.push(("wav:rf64.data_size".to_string(), data_size.to_string()));
    out.push((
        "wav:rf64.sample_count".to_string(),
        sample_count.to_string(),
    ));
    out.push(("wav:rf64.table.count".to_string(), table_len.to_string()));

    const REC_LEN: usize = 12;
    let mut table = Vec::with_capacity(table_len);
    let table_bytes_available = buf.len().saturating_sub(28);
    let table_recs_available = table_bytes_available / REC_LEN;
    // Defensive vs. writers that lie about the count: only consume as
    // many records as the body actually carries.
    let n = table_len.min(table_recs_available);
    for i in 0..n {
        let off = 28 + i * REC_LEN;
        let id: [u8; 4] = [buf[off], buf[off + 1], buf[off + 2], buf[off + 3]];
        let size = u64::from_le_bytes([
            buf[off + 4],
            buf[off + 5],
            buf[off + 6],
            buf[off + 7],
            buf[off + 8],
            buf[off + 9],
            buf[off + 10],
            buf[off + 11],
        ]);
        let id_str = if id.iter().all(|&b| (0x20..=0x7E).contains(&b)) {
            String::from_utf8_lossy(&id).to_string()
        } else {
            format!("0x{:02X}{:02X}{:02X}{:02X}", id[0], id[1], id[2], id[3])
        };
        out.push((format!("wav:rf64.table.{i}.id"), id_str.clone()));
        out.push((format!("wav:rf64.table.{i}.size"), size.to_string()));
        table.push((id, size));
    }
    out.push(("wav:rf64.body_len".to_string(), buf.len().to_string()));
    Ok(Ds64 {
        riff_size,
        data_size,
        sample_count,
        table,
    })
}

/// Resolve a chunk's real size: when the 32-bit on-wire size is the
/// `0xFFFFFFFF` sentinel, look up the 64-bit override in the `ds64`
/// table (matching FOURCC). For chunks that aren't in the table the
/// sentinel cannot be resolved — returns `None` so the caller can
/// reject the file as malformed rather than silently skipping forward
/// by a billion bytes.
fn resolve_chunk_size(id: &[u8; 4], on_wire: u32, ds64: Option<&Ds64>) -> Option<u64> {
    if on_wire != SIZE64_SENTINEL {
        return Some(on_wire as u64);
    }
    let ds64 = ds64?;
    // `data` size override lives in the dedicated `dataSize` field
    // rather than the table.
    if id == b"data" {
        return Some(ds64.data_size);
    }
    ds64.table
        .iter()
        .find(|(tid, _)| tid == id)
        .map(|(_, sz)| *sz)
}

// On-the-wire `wFormatTag` constants from RFC 2361 / `mmreg.h`. Public so
// muxer callers can build `WAVE_FORMAT_EXTENSIBLE` streams against the
// same dispatch table the demuxer uses.
/// `WAVE_FORMAT_PCM` — integer linear PCM (`mmreg.h`).
pub const WAVE_FORMAT_PCM: u16 = 0x0001;
/// `WAVE_FORMAT_IEEE_FLOAT` — 32-bit / 64-bit IEEE 754 float PCM.
pub const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
/// `WAVE_FORMAT_ALAW` — ITU-T G.711 A-law (RFC 2361 A.7).
pub const WAVE_FORMAT_ALAW: u16 = 0x0006;
/// `WAVE_FORMAT_MULAW` — ITU-T G.711 μ-law (RFC 2361 A.8).
pub const WAVE_FORMAT_MULAW: u16 = 0x0007;
/// `WAVE_FORMAT_EXTENSIBLE` — escape hatch with 22-byte extension
/// carrying `wValidBitsPerSample` / `dwChannelMask` / `SubFormat` GUID
/// (per docs/container/riff/waveformatextensible/README.md).
pub const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

// Internal aliases kept for readability of the local match arms below.
const FMT_PCM: u16 = WAVE_FORMAT_PCM;
const FMT_IEEE_FLOAT: u16 = WAVE_FORMAT_IEEE_FLOAT;
const FMT_ALAW: u16 = WAVE_FORMAT_ALAW;
const FMT_MULAW: u16 = WAVE_FORMAT_MULAW;
const FMT_EXTENSIBLE: u16 = WAVE_FORMAT_EXTENSIBLE;

// `KSDATAFORMAT_SUBTYPE_*` GUIDs (`KSMedia.h`). All follow the same
// `<tag>-0000-0010-8000-00AA00389B71` "DataFormat" base where the
// leading 16-bit `<tag>` is the legacy `wFormatTag` — per
// docs/container/riff/waveformatextensible/README.md.
const GUID_PCM: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
const GUID_IEEE_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
const GUID_ALAW: [u8; 16] = [
    0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
const GUID_MULAW: [u8; 16] = [
    0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// The 14 trailing bytes of `KSDATAFORMAT_SUBTYPE_WAVEFORMATEX`
/// (`{00000000-0000-0010-8000-00aa00389b71}`) — i.e. the GUID with its
/// leading 16-bit `Data1` field (the embedded `wFormatTag`) removed.
///
/// Per
/// `docs/container/riff/waveformatextensible/ms-converting-format-tags-and-subformat-guids.md`,
/// the `KSMedia.h` macro `DEFINE_WAVEFORMATEX_GUID(x)` constructs a
/// SubFormat GUID as `(USHORT)(x), 0x0000, 0x0010, 0x80, 0x00, 0x00,
/// 0xaa, 0x00, 0x38, 0x9b, 0x71` — the legacy `wFormatTag` `x` occupies
/// the low 16 bits of `Data1`, the high 16 bits of `Data1` are zero, and
/// the remaining twelve bytes are the fixed "WAVEFORMATEX" base. The
/// companion `IS_VALID_WAVEFORMATEX_GUID(Guid)` macro tests a candidate
/// GUID by comparing every byte *after* the leading `USHORT` (i.e. bytes
/// `[2..16]`) against this base; `EXTRACT_WAVEFORMATEX_ID(Guid)` then
/// reads the legacy tag from `(USHORT)(Guid->Data1)` (bytes `[0..2]`,
/// little-endian).
const GUID_WAVEFORMATEX_TAIL: [u8; 14] = [
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// If `g` is a SubFormat GUID produced by the `KSMedia.h`
/// `DEFINE_WAVEFORMATEX_GUID(x)` template (a "WAVEFORMATEX GUID"),
/// return the embedded legacy `wFormatTag` `x`; otherwise `None`.
///
/// Implements `IS_VALID_WAVEFORMATEX_GUID` + `EXTRACT_WAVEFORMATEX_ID`
/// from
/// `docs/container/riff/waveformatextensible/ms-converting-format-tags-and-subformat-guids.md`:
/// the validity test compares the fourteen bytes after the leading
/// `Data1` low half — bytes `[2..16]` — against the fixed
/// `KSDATAFORMAT_SUBTYPE_WAVEFORMATEX` tail (which includes the zero high
/// half of `Data1` at byte `[2..4]`), and the tag is the little-endian
/// `u16` at bytes `[0..2]`.
fn waveformatex_tag(g: &[u8; 16]) -> Option<u16> {
    if g[2..16] == GUID_WAVEFORMATEX_TAIL {
        Some(u16::from_le_bytes([g[0], g[1]]))
    } else {
        None
    }
}

/// Format the 16-byte SubFormat GUID for diagnostic strings as
/// `XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX` (canonical text representation
/// used by `mmreg.h` GUID definitions). The first three groups are
/// little-endian on the wire, the trailing two groups are big-endian.
fn fmt_guid(g: &[u8; 16]) -> String {
    format!(
        "{:08X}-{:04X}-{:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
        u32::from_le_bytes([g[0], g[1], g[2], g[3]]),
        u16::from_le_bytes([g[4], g[5]]),
        u16::from_le_bytes([g[6], g[7]]),
        g[8],
        g[9],
        g[10],
        g[11],
        g[12],
        g[13],
        g[14],
        g[15],
    )
}

/// `WAVEFORMATEXTENSIBLE.dwChannelMask` `SPEAKER_*` flag bits, ordered
/// least-significant-bit first. Per
/// `docs/container/riff/waveformatextensible/ms-waveformatextensible.html`
/// §"dwChannelMask": the LSB is the front-left speaker, the next bit the
/// front-right speaker, and so on through bit 17 (`SPEAKER_TOP_BACK_RIGHT`,
/// `0x20000`). The interleaved PCM samples appear in this same
/// least-significant-bit-up order.
const SPEAKER_FLAGS: [(u32, &str); 18] = [
    (0x1, "FRONT_LEFT"),
    (0x2, "FRONT_RIGHT"),
    (0x4, "FRONT_CENTER"),
    (0x8, "LOW_FREQUENCY"),
    (0x10, "BACK_LEFT"),
    (0x20, "BACK_RIGHT"),
    (0x40, "FRONT_LEFT_OF_CENTER"),
    (0x80, "FRONT_RIGHT_OF_CENTER"),
    (0x100, "BACK_CENTER"),
    (0x200, "SIDE_LEFT"),
    (0x400, "SIDE_RIGHT"),
    (0x800, "TOP_CENTER"),
    (0x1000, "TOP_FRONT_LEFT"),
    (0x2000, "TOP_FRONT_CENTER"),
    (0x4000, "TOP_FRONT_RIGHT"),
    (0x8000, "TOP_BACK_LEFT"),
    (0x10000, "TOP_BACK_CENTER"),
    (0x20000, "TOP_BACK_RIGHT"),
];

/// Decode a `WAVEFORMATEXTENSIBLE.dwChannelMask` bitmap into a
/// human-readable, `+`-separated list of `SPEAKER_*` positions in the
/// canonical least-significant-bit-first order
/// (`FRONT_LEFT+FRONT_RIGHT+...`).
///
/// Returns `None` when `mask == 0` (no assigned speaker positions —
/// "use direct-out / discrete channels"). Any bits set above the highest
/// defined flag (`0x20000`) that aren't recognised are reported as
/// `UNKNOWN(0x...)` so the round-trip information isn't silently dropped.
///
/// Per
/// `docs/container/riff/waveformatextensible/ms-waveformatextensible.html`,
/// the number of set bits should equal `WAVEFORMATEX.nChannels`; this
/// function only decodes the mask and does not enforce that invariant.
fn channel_mask_layout(mask: u32) -> Option<String> {
    if mask == 0 {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    let mut known: u32 = 0;
    for (bit, name) in SPEAKER_FLAGS {
        if mask & bit != 0 {
            parts.push(name.to_string());
            known |= bit;
        }
    }
    let unknown = mask & !known;
    if unknown != 0 {
        parts.push(format!("UNKNOWN(0x{unknown:X})"));
    }
    Some(parts.join("+"))
}

// --- Demuxer ---------------------------------------------------------------

fn open_demuxer(input: Box<dyn ReadSeek>, _codecs: &dyn CodecResolver) -> Result<Box<dyn Demuxer>> {
    Ok(Box::new(open_wav_demuxer(input)?))
}

/// Open a WAV/RF64/BW64 demuxer returning the concrete [`WavDemuxer`]
/// so the typed accessor surface ([`WavDemuxer::format_tag`],
/// [`WavDemuxer::channel_mask`], [`WavDemuxer::acid`], …) is reachable
/// without downcasting. The registry path wraps this in a
/// `Box<dyn Demuxer>`.
pub fn open_wav_demuxer(mut input: Box<dyn ReadSeek>) -> Result<WavDemuxer> {
    let mut hdr = [0u8; 12];
    input.read_exact(&mut hdr)?;
    let magic: [u8; 4] = [hdr[0], hdr[1], hdr[2], hdr[3]];
    let is_rf64 = &magic == b"RF64";
    let is_bw64 = &magic == b"BW64";
    let is_riff = &magic == b"RIFF";
    if !(is_riff || is_rf64 || is_bw64) || &hdr[8..12] != b"WAVE" {
        return Err(Error::invalid("not a RIFF/RF64/BW64 WAVE file"));
    }

    // Walk chunks until we hit "data"; parse "fmt ", "ds64" and "LIST"
    // along the way.
    let mut fmt: Option<WaveFmt> = None;
    let mut metadata: Vec<(String, String)> = Vec::new();
    // `fact` chunk's `dwFileSize` (per-channel sample count) when present.
    // Required for non-PCM / `wavl LIST` waveform data per
    // `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "FACT
    // Chunk" — the only honest sample count when `block_align *
    // total_samples != data_size`.
    let mut fact_sample_count: Option<u64> = None;
    // Typed Acidizer view, populated when an `acid` chunk parses.
    let mut acid: Option<AcidChunk> = None;
    // Typed BW64/ADM channel-allocation view, populated when a `chna`
    // chunk parses (ITU-R BS.2088-2 §8.1).
    let mut chna: Option<ChnaChunk> = None;
    // Typed BWF Broadcast Audio Extension view, populated when a `bext`
    // chunk parses (EBU Tech 3285 v2 §2.3).
    let mut bext: Option<BextChunk> = None;
    // Typed cue-points / playlist / associated-data-list views, populated
    // from the `cue `, `plst` and `LIST adtl` chunks (RIFF MCI §3). The
    // first well-formed occurrence of each wins.
    let mut cue: Option<CueChunk> = None;
    let mut plst: Option<PlaylistChunk> = None;
    let mut adtl: Option<AdtlChunk> = None;
    // RF64/BW64 ds64 (EBU Tech 3306 §3 / Annex A.2): mandatory first
    // chunk after the form header when the magic is RF64 or BW64;
    // otherwise must be absent.
    let mut ds64: Option<Ds64> = None;
    if is_rf64 || is_bw64 {
        // Surface the form-magic up front so a downstream tool can
        // distinguish the three identical-otherwise top-level shapes
        // without re-reading the input.
        let form = if is_rf64 { "RF64" } else { "BW64" };
        metadata.push(("wav:rf64.magic".to_string(), form.to_string()));
        // The legacy 32-bit "RIFF size" field at offset 4..8 of the
        // form header must be the sentinel for a well-formed RF64 /
        // BW64 file (EBU Tech 3306 §3 last paragraph + Annex A.2
        // RF64Chunk comment). We surface the on-wire value so a
        // non-compliant writer is observable but don't reject.
        let on_wire_riff = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        if on_wire_riff != SIZE64_SENTINEL {
            metadata.push(("wav:rf64.riff_size32".to_string(), on_wire_riff.to_string()));
        }
    }
    let mut data_offset: Option<u64> = None;
    let mut data_size: Option<u64> = None;
    loop {
        let mut chdr = [0u8; 8];
        // A `wavl`-form file has no top-level `data` chunk to break on, so
        // the scan walks every remaining chunk and terminates at a clean
        // end-of-stream. A short/absent header at EOF after a `wavl` LIST
        // already anchored the cursor is the normal termination; only an
        // EOF before any waveform is anchored is malformed (caught below).
        match input.read_exact(&mut chdr) {
            Ok(()) => {}
            Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let id_arr: [u8; 4] = [chdr[0], chdr[1], chdr[2], chdr[3]];
        let id = &chdr[0..4];
        let on_wire_size = u32::from_le_bytes([chdr[4], chdr[5], chdr[6], chdr[7]]);
        // ds64-aware size resolution: only `data` and the table-listed
        // chunk-IDs can carry the 32-bit sentinel; the standalone
        // chunks (`fmt `, `bext`, etc.) are bounded by RIFF's 32-bit
        // arithmetic and never hit the sentinel.
        let size = match resolve_chunk_size(&id_arr, on_wire_size, ds64.as_ref()) {
            Some(s) => s,
            None => {
                return Err(Error::invalid(format!(
                    "RF64 chunk {:?} carries 0xFFFFFFFF sentinel but no ds64 override",
                    String::from_utf8_lossy(id)
                )));
            }
        };
        match id {
            b"ds64" => {
                // RF64/BW64: mandatory, must precede `fmt `/`data`. For
                // a plain RIFF file the chunk must not appear; if it
                // does we surface the keys but otherwise treat it as
                // an unknown ahead-of-fmt extension and keep going.
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                let parsed = parse_ds64_chunk(&buf, &mut metadata)?;
                ds64 = Some(parsed);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"fmt " => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                fmt = Some(parse_fmt(&buf)?);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"fact" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                // RF64/BW64: when the 32-bit `dwFileSize` carries the
                // sentinel, the authoritative 64-bit sample count
                // lives in `ds64.sampleCount` (EBU Tech 3306 §3 last
                // bullet of the three-mandatory-fields list). The
                // standard `parse_fact_chunk` only returns the 32-bit
                // legacy field; we promote here when needed.
                let legacy = parse_fact_chunk(&buf, &mut metadata);
                fact_sample_count = match legacy {
                    Some(v) if v == SIZE64_SENTINEL => ds64.as_ref().map(|d| d.sample_count),
                    Some(v) => Some(v as u64),
                    None => None,
                };
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"LIST" => {
                // The LIST chunk body opens with a 4-byte list type. The
                // `wavl` (wave-list) type per Microsoft RIFF MCI §3
                // "Storage of WAVE Data" is the segmented waveform
                // container —
                // `LIST('wavl' { <data-ck> | <silence-ck> }... )` —
                // alternating `data` payloads with `slnt` silence
                // counts. We need byte offsets for the embedded `data`
                // sub-chunks, so capture the LIST body's absolute start
                // and resolve the first `data` segment as the decode
                // anchor; `INFO` / `adtl` LISTs are pure metadata and
                // stay on the buffered path.
                let list_start = input.stream_position()?;
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                if buf.len() >= 4 && &buf[0..4] == b"wavl" {
                    if let Some((off, sz)) = parse_wavl_list(&buf, list_start, &mut metadata) {
                        // The §3 grammar lets `<wave-data>` be a `wavl`
                        // LIST instead of a top-level `data` chunk. Anchor
                        // the decode cursor at the first `data` segment so
                        // the leading audio is readable; later segments
                        // (and embedded silence) are surfaced as
                        // `wav:wavl.*` metadata for downstream walking.
                        if data_offset.is_none() {
                            data_offset = Some(off);
                            data_size = Some(sz);
                        }
                    }
                } else {
                    if buf.len() >= 4 && &buf[0..4] == b"adtl" && adtl.is_none() {
                        let parsed = AdtlChunk::parse(&buf[4..]);
                        if !parsed.entries.is_empty() {
                            adtl = Some(parsed);
                        }
                    }
                    parse_list_chunk(&buf, &mut metadata);
                }
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"bext" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                bext = parse_bext_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"cue " => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_cue_chunk(&buf, &mut metadata);
                if cue.is_none() {
                    cue = CueChunk::parse(&buf);
                }
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"plst" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_plst_chunk(&buf, &mut metadata);
                if plst.is_none() {
                    plst = PlaylistChunk::parse(&buf);
                }
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"smpl" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_smpl_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"inst" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_inst_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"acid" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                acid = parse_acid_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"chna" => {
                // ITU-R BS.2088-2 §8.1: the BW64/ADM channel-allocation
                // chunk maps each track in the `data` interleave to its
                // ADM audioTrackUID / audioTrackFormatID /
                // audioPackFormatID references. Body is a 4-byte count
                // pre-amble followed by N fixed 40-byte `audioID`
                // records.
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                chna = parse_chna_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"iXML" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_ixml_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"axml" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_axml_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"bxml" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_bxml_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"_PMX" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_pmx_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"CSET" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_cset_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"JUNK" => {
                // Microsoft RIFF MCI §2 "JUNK (Filler) Chunk": padding,
                // filler or outdated information; the body contains
                // random data and carries no relevant payload. We skip
                // the body but surface accounting metadata so a
                // downstream tool can observe how much filler the
                // producer reserved (e.g. for in-place editing) and
                // how many JUNK chunks appeared. Multiple JUNK chunks
                // are allowed; the count/total-bytes accumulate.
                input.seek(SeekFrom::Current(size as i64))?;
                surface_junk_metadata(&mut metadata, size);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"slnt" => {
                // Microsoft RIFF MCI §3 "Wave Data": the `slnt`
                // (silence) chunk `slnt( <dwSamples:DWORD> )` carries a
                // single DWORD count of silent samples rather than a
                // stretch of zeroed sample data. It normally appears
                // inside a `wavl` LIST alternating with `data` chunks,
                // but the spec allows it as a sibling of `data` at the
                // top level too. We surface its sample count without
                // synthesising real silence into the decoded stream.
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                surface_slnt_metadata(&mut metadata, &buf);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"data" => {
                // The `data` chunk is no longer the scan terminator:
                // cue / plst / LIST adtl chunks are commonly placed
                // *after* the waveform (RIFF MCI §3 lists them among the
                // optional `<other-ck>` which may follow `data`). Record
                // the payload anchor on the first `data` chunk, then seek
                // over its (word-aligned) body and keep walking so the
                // trailing metadata chunks are parsed too.
                if data_offset.is_none() {
                    data_offset = Some(input.stream_position()?);
                    data_size = Some(size);
                }
                let pad = size + (size % 2);
                input.seek(SeekFrom::Current(pad as i64))?;
            }
            _ => {
                let pad = size + (size % 2);
                input.seek(SeekFrom::Current(pad as i64))?;
            }
        }
    }
    let fmt = fmt.ok_or_else(|| Error::invalid("WAV missing fmt chunk"))?;
    // Either a top-level `data` chunk (the common case) or the first
    // `data` sub-chunk of a `wavl` LIST must have anchored the cursor.
    let data_offset =
        data_offset.ok_or_else(|| Error::invalid("WAV missing data / wavl waveform"))?;
    let data_size = data_size.unwrap_or(0);

    let codec_id = resolve_codec(&fmt)?;
    // Sample format hint for the decoded shape (NOT the on-wire layout
    // — A-law/μ-law expand to S16 once decoded, mirroring the
    // round-75 `oxideav-avi` audio_codec_sample_format helper). For
    // synthesised `wav:guid_<...>` ids the decoded shape is unknown
    // so we leave sample_format as None — callers can still pull
    // channels / sample_rate / channel_mask / SubFormat to surface a
    // useful diagnostic without us pretending to know the codec.
    let sample_fmt = decoded_sample_format(&codec_id);

    let time_base = TimeBase::new(1, fmt.sample_rate as i64);
    let block_align = fmt.block_align.max(1) as u64;
    let block_samples = data_size / block_align;
    // Prefer the `fact` chunk's per-channel sample count when present —
    // for compressed WAV streams (and for `wavl LIST` containers in
    // general) the `data_size / block_align` heuristic is meaningless
    // because one byte of payload no longer maps to one sample. For PCM
    // the two should agree; when they don't we surface
    // `wav:fact.mismatch` so a downstream tool can flag the file rather
    // than silently trusting one number over the other.
    let total_samples = if let Some(fc) = fact_sample_count {
        if fc != block_samples {
            metadata.push((
                "wav:fact.mismatch".to_string(),
                format!("block_samples={block_samples} fact_samples={fc}"),
            ));
        }
        fc
    } else {
        block_samples
    };
    let duration_micros: i64 = if fmt.sample_rate > 0 {
        (total_samples as i128 * 1_000_000 / fmt.sample_rate as i128) as i64
    } else {
        0
    };

    let mut params = CodecParameters::audio(codec_id);
    params.tag = Some(oxideav_core::CodecTag::wave_format(fmt.format_tag));
    params.channels = Some(fmt.channels);
    params.sample_rate = Some(fmt.sample_rate);
    params.sample_format = sample_fmt;
    // bit_rate uses the on-wire bytes_per_second (== block_align *
    // sample_rate) — for A-law/μ-law that's 8 * channels * rate, NOT
    // the post-decode S16 rate.
    params.bit_rate = Some(8 * block_align * (fmt.sample_rate as u64));

    // Round-77 metadata: surface WAVEFORMATEXTENSIBLE side-info under
    // the same key shape `oxideav-avi` uses (without the per-stream
    // index — WAV is single-stream by construction). Only emitted when
    // the on-wire wFormatTag is EXTENSIBLE and the extension parsed.
    if fmt.format_tag == FMT_EXTENSIBLE {
        if let Some(valid) = fmt.valid_bits_per_sample {
            metadata.push((
                "wav:fmt.valid_bits_per_sample".to_string(),
                valid.to_string(),
            ));
        }
        if let Some(mask) = fmt.channel_mask {
            metadata.push(("wav:fmt.channel_mask".to_string(), format!("0x{mask:08X}")));
            if let Some(layout) = channel_mask_layout(mask) {
                metadata.push(("wav:fmt.channel_layout".to_string(), layout));
            }
        }
        if let Some(sub) = &fmt.subformat {
            metadata.push(("wav:fmt.subformat".to_string(), fmt_guid(sub)));
            // When the GUID follows the KSMedia.h DEFINE_WAVEFORMATEX_GUID
            // template, surface the embedded legacy wFormatTag it is
            // equivalent to (per
            // docs/.../ms-converting-format-tags-and-subformat-guids.md).
            // Lets a downstream tool see that e.g. an EXTENSIBLE file with
            // subformat {00000055-...} is MP3-tagged without re-deriving
            // the mapping.
            if let Some(tag) = waveformatex_tag(sub) {
                metadata.push(("wav:fmt.subformat_tag".to_string(), format!("0x{tag:04X}")));
            }
        }
    }

    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(total_samples as i64),
        start_time: Some(0),
        params,
    };

    Ok(WavDemuxer {
        input,
        streams: vec![stream],
        data_offset,
        data_end: data_offset + data_size,
        cursor: data_offset,
        block_align,
        chunk_frames: 1024,
        samples_emitted: 0,
        metadata,
        duration_micros,
        format_tag: fmt.format_tag,
        valid_bits_per_sample: fmt.valid_bits_per_sample,
        channel_mask: fmt.channel_mask,
        subformat: fmt.subformat,
        acid,
        chna,
        bext,
        cue,
        plst,
        adtl,
    })
}

/// Parse a RIFF LIST chunk body. Dispatches by the 4-byte list type:
/// `INFO` maps `IART`/`INAM`/... sub-chunks to standard key names
/// (`artist`, `title`, ...); `adtl` (Associated Data List, per
/// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "Associated
/// Data Chunk") maps `labl`/`note`/`ltxt`/`file` sub-chunks to
/// `wav:adtl.<id>.<dwName>` keys carrying the cue-point text /
/// embedded-file accounting.
fn parse_list_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    if buf.len() < 4 {
        return;
    }
    match &buf[0..4] {
        b"INFO" => parse_info_list(&buf[4..], out),
        b"adtl" => parse_adtl_list(&buf[4..], out),
        _ => {}
    }
}

/// Parse a `LIST('wavl' ...)` wave-list body per Microsoft RIFF MCI §3
/// "Storage of WAVE Data":
///
/// > `<wave-data>` ➝ `{ <data-ck> | <data-list> }`
/// > `<wave-list>` ➝ `LIST( 'wavl' { <data-ck> | <silence-ck> }... )`
/// > `<silence-ck>` ➝ `slnt( <dwSamples:DWORD> )`
///
/// The `wavl` form interleaves runs of real PCM (`data` sub-chunks) with
/// `slnt` silence-count markers, letting a writer encode long silent
/// stretches sparsely instead of storing zeroed samples. The §3 note is
/// explicit that `slnt` is a *count of silent samples*, not a baseline
/// fill, so we do not synthesise samples — silence is surfaced through
/// the same `wav:slnt.*` accounting used for top-level `slnt` chunks.
///
/// `buf` is the LIST body (starting at the `wavl` list type); `list_start`
/// is the absolute stream offset of that first body byte, so the returned
/// `(offset, size)` of the first `data` sub-chunk is an absolute file
/// offset the demuxer can seek to. Returns `None` when the LIST holds no
/// `data` segment (a silence-only `wavl`, which carries no decodable
/// audio but is still fully surfaced as metadata).
///
/// Surfaced keys:
/// * `wav:wavl.segment_count` — total `data` + `slnt` sub-chunks walked.
/// * `wav:wavl.data_count` — number of `data` segments.
/// * `wav:wavl.data_bytes` — cumulative payload bytes across `data`
///   segments (excludes the 8-byte sub-chunk headers and word-align
///   padding).
/// * `wav:wavl.<n>.kind` / `.length` — per-segment type (`data`/`slnt`)
///   and on-wire body length, indexed zero-based by encounter order.
/// * embedded `slnt` segments additionally feed `wav:slnt.*` so the
///   silent-sample totals match a top-level-`slnt` file.
fn parse_wavl_list(
    buf: &[u8],
    list_start: u64,
    out: &mut Vec<(String, String)>,
) -> Option<(u64, u64)> {
    // Skip the 4-byte 'wavl' list type.
    let mut i = 4usize;
    let mut first_data: Option<(u64, u64)> = None;
    let mut segment_count: u64 = 0;
    let mut data_count: u64 = 0;
    let mut data_bytes: u64 = 0;
    while i + 8 <= buf.len() {
        let id: [u8; 4] = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
        let size = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
        let body = i + 8;
        if body + size > buf.len() {
            break;
        }
        let idx = segment_count;
        match &id {
            b"data" => {
                out.push((format!("wav:wavl.{idx}.kind"), "data".to_string()));
                out.push((format!("wav:wavl.{idx}.length"), size.to_string()));
                data_count += 1;
                data_bytes = data_bytes.saturating_add(size as u64);
                if first_data.is_none() {
                    // Absolute offset = LIST body start + sub-chunk body
                    // offset within the body.
                    first_data = Some((list_start + body as u64, size as u64));
                }
            }
            b"slnt" => {
                out.push((format!("wav:wavl.{idx}.kind"), "slnt".to_string()));
                out.push((format!("wav:wavl.{idx}.length"), size.to_string()));
                surface_slnt_metadata(out, &buf[body..body + size]);
            }
            // The §3 grammar admits only `data` and `slnt` inside `wavl`;
            // anything else is an unknown forward extension — record its
            // presence so the file stays observable, but don't anchor on
            // it.
            _ => {
                out.push((
                    format!("wav:wavl.{idx}.kind"),
                    String::from_utf8_lossy(&id).trim().to_string(),
                ));
                out.push((format!("wav:wavl.{idx}.length"), size.to_string()));
            }
        }
        segment_count += 1;
        // RIFF word-alignment: sub-chunks are padded to an even length.
        i = body + size + (size & 1);
    }
    out.push((
        "wav:wavl.segment_count".to_string(),
        segment_count.to_string(),
    ));
    out.push(("wav:wavl.data_count".to_string(), data_count.to_string()));
    out.push(("wav:wavl.data_bytes".to_string(), data_bytes.to_string()));
    first_data
}

fn parse_info_list(buf: &[u8], out: &mut Vec<(String, String)>) {
    let mut i = 0usize;
    while i + 8 <= buf.len() {
        let id: [u8; 4] = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
        let size = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
        i += 8;
        if i + size > buf.len() {
            break;
        }
        let raw = &buf[i..i + size];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let value = String::from_utf8_lossy(&raw[..end]).trim().to_string();
        let key = info_id_to_key(&id);
        if !value.is_empty() {
            if let Some(k) = key {
                out.push((k.to_string(), value));
            }
        }
        i += size;
        if size % 2 == 1 {
            i += 1;
        }
    }
}

/// Parse a `LIST adtl` (Associated Data List) body and emit
/// `wav:adtl.<sub-id>.<dwName>` keys carrying the text payload of each
/// sub-chunk. Per `docs/container/riff/metadata/microsoft-riffmci.pdf`
/// §3 "Associated Data Chunk":
///
/// * `labl(<dwName:DWORD> <data:ZSTR>)` — title text for cue `dwName`.
/// * `note(<dwName:DWORD> <data:ZSTR>)` — comment text for cue `dwName`.
/// * `ltxt(<dwName> <dwSampleLength> <dwPurpose> <wCountry> <wLanguage>
///   <wDialect> <wCodePage> <data:BYTE>...)` — text covering a
///   `dwSampleLength`-sample segment starting at cue `dwName`. The
///   parser surfaces the segment length under `.ltxt.<dwName>.length`,
///   the FOURCC purpose under `.ltxt.<dwName>.purpose`, the text
///   payload (trimmed at the first NUL) under `.ltxt.<dwName>.text`,
///   and the four locale WORDs under `.ltxt.<dwName>.country` /
///   `.language` / `.dialect` / `.code_page` (raw decimals, always
///   emitted). Per §3 "Text with Data Length Information" the country
///   and `(language, dialect)` codes come from the same Chapter-2
///   tables the `CSET` chunk uses, so the parser resolves them through
///   the shared table lookups into `.ltxt.<dwName>.country_name` /
///   `.language_name` (emitted only when the code is in the spec's
///   enumerated set).
/// * `file(<dwName:DWORD> <dwMedType:DWORD> <fileData:BYTE>...)` —
///   embedded media file for cue `dwName`. Per §3 "Embedded File
///   Information" `dwMedType` identifies the file type carried in
///   `fileData` ("If the fileData section contains a RIFF form, the
///   dwMedType field is the same as the RIFF form type"; zero is
///   explicitly allowed). The parser surfaces the type under
///   `.file.<dwName>.med_type` (FOURCC text when printable, `0` for
///   the spec-allowed zero value, hex otherwise) and the embedded
///   payload length under `.file.<dwName>.body_len`. The `fileData`
///   bytes themselves are not surfaced through the string-typed
///   metadata API — `body_len` keeps the payload observable without
///   pretending the parser can interpret the inner format.
///
/// Sub-chunks shorter than the minimum required header are skipped.
fn parse_adtl_list(buf: &[u8], out: &mut Vec<(String, String)>) {
    let mut i = 0usize;
    while i + 8 <= buf.len() {
        let id: [u8; 4] = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
        let size = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
        i += 8;
        if i + size > buf.len() {
            break;
        }
        let body = &buf[i..i + size];
        match &id {
            b"labl" | b"note" if body.len() >= 4 => {
                let dw_name = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);
                let raw = &body[4..];
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                let text = String::from_utf8_lossy(&raw[..end]).trim().to_string();
                if !text.is_empty() {
                    let sub = if &id == b"labl" { "labl" } else { "note" };
                    out.push((format!("wav:adtl.{sub}.{dw_name}"), text));
                }
            }
            // 4 dwName + 4 dwSampleLength + 4 dwPurpose + 2 wCountry
            // + 2 wLanguage + 2 wDialect + 2 wCodePage = 20 bytes
            // fixed header.
            b"ltxt" if body.len() >= 20 => {
                let dw_name = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);
                let dw_length = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                let purpose = &body[8..12];
                out.push((
                    format!("wav:adtl.ltxt.{dw_name}.length"),
                    dw_length.to_string(),
                ));
                // Render purpose as a FOURCC when printable, hex otherwise.
                let purpose_str = if purpose.iter().all(|&b| (0x20..=0x7E).contains(&b)) {
                    String::from_utf8_lossy(purpose).to_string()
                } else {
                    format!(
                        "0x{:02X}{:02X}{:02X}{:02X}",
                        purpose[0], purpose[1], purpose[2], purpose[3]
                    )
                };
                out.push((format!("wav:adtl.ltxt.{dw_name}.purpose"), purpose_str));
                // §3 "Text with Data Length Information": wCountry,
                // (wLanguage, wDialect) and wCodePage qualify the text
                // payload, drawing on the same Chapter-2 country /
                // language-and-dialect tables that the CSET chunk uses.
                // Raw decimals are always emitted (zero = "use the
                // default" per the CSET zero-value semantics); the
                // human-readable names only when the table resolves.
                let country = u16::from_le_bytes([body[12], body[13]]);
                let language = u16::from_le_bytes([body[14], body[15]]);
                let dialect = u16::from_le_bytes([body[16], body[17]]);
                let code_page = u16::from_le_bytes([body[18], body[19]]);
                out.push((
                    format!("wav:adtl.ltxt.{dw_name}.country"),
                    country.to_string(),
                ));
                if let Some(name) = cset_country_name(country) {
                    out.push((
                        format!("wav:adtl.ltxt.{dw_name}.country_name"),
                        name.to_string(),
                    ));
                }
                out.push((
                    format!("wav:adtl.ltxt.{dw_name}.language"),
                    language.to_string(),
                ));
                out.push((
                    format!("wav:adtl.ltxt.{dw_name}.dialect"),
                    dialect.to_string(),
                ));
                if let Some(name) = cset_language_name(language, dialect) {
                    out.push((
                        format!("wav:adtl.ltxt.{dw_name}.language_name"),
                        name.to_string(),
                    ));
                }
                out.push((
                    format!("wav:adtl.ltxt.{dw_name}.code_page"),
                    code_page.to_string(),
                ));
                let raw = &body[20..];
                // The text payload may or may not be NUL-terminated
                // per the spec ("<data:BYTE>..."); trim at the first
                // NUL if present and strip surrounding whitespace.
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                let text = String::from_utf8_lossy(&raw[..end]).trim().to_string();
                if !text.is_empty() {
                    out.push((format!("wav:adtl.ltxt.{dw_name}.text"), text));
                }
            }
            // §3 "Embedded File Information": dwName + dwMedType fixed
            // header, then the embedded file bytes. The fileData payload
            // is not surfaced through the string-typed metadata API;
            // `med_type` + `body_len` keep it observable.
            b"file" if body.len() >= 8 => {
                let dw_name = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);
                let med_type = &body[4..8];
                // "This field can contain a zero value" — render the
                // spec-allowed zero as plain `0`, a printable FOURCC as
                // text, anything else as hex.
                let med_type_str = if med_type == [0, 0, 0, 0] {
                    "0".to_string()
                } else if med_type.iter().all(|&b| (0x20..=0x7E).contains(&b)) {
                    String::from_utf8_lossy(med_type).to_string()
                } else {
                    format!(
                        "0x{:02X}{:02X}{:02X}{:02X}",
                        med_type[0], med_type[1], med_type[2], med_type[3]
                    )
                };
                out.push((format!("wav:adtl.file.{dw_name}.med_type"), med_type_str));
                out.push((
                    format!("wav:adtl.file.{dw_name}.body_len"),
                    (body.len() - 8).to_string(),
                ));
            }
            // Truncated `labl`/`note`/`ltxt`/`file` (under the
            // fixed-header minimum) are skipped as opaque.
            _ => {}
        }
        i += size;
        if size % 2 == 1 {
            i += 1;
        }
    }
}

/// Parse a `cue ` chunk body and emit `wav:cue.count` + per-point
/// `wav:cue.<dwName>.position` / `.fcc_chunk` / `.chunk_start` /
/// `.block_start` / `.sample_offset` keys. Layout per
/// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "Cue-Points
/// Chunk":
///
/// ```text
/// <cue-ck> -> cue( <dwCuePoints:DWORD> <cue-point>... )
/// <cue-point> -> struct {
///     DWORD dwName;        // unique id, referenced by plst/adtl
///     DWORD dwPosition;    // sample position in the play order
///     FOURCC fccChunk;     // 'data' or 'slnt' (for wavl LIST forms)
///     DWORD dwChunkStart;  // byte offset of fccChunk within wavl LIST
///     DWORD dwBlockStart;  // byte offset of enclosing block
///     DWORD dwSampleOffset;// sample offset within block
/// }
/// ```
///
/// A truncated chunk (count > body) is treated as opaque and skipped.
/// Each cue-point record is 24 bytes; the function consumes as many
/// records as the body actually carries even if `dwCuePoints` claims
/// more (defensive vs. writers that lie about the count).
fn parse_cue_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    if buf.len() < 4 {
        return;
    }
    let count_claimed = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let body = &buf[4..];
    const REC_LEN: usize = 24;
    let count_actual = (body.len() / REC_LEN) as u32;
    let count = count_claimed.min(count_actual);
    out.push(("wav:cue.count".to_string(), count.to_string()));
    for i in 0..count as usize {
        let off = i * REC_LEN;
        let dw_name = u32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]);
        let dw_position =
            u32::from_le_bytes([body[off + 4], body[off + 5], body[off + 6], body[off + 7]]);
        let fcc_chunk = &body[off + 8..off + 12];
        let dw_chunk_start = u32::from_le_bytes([
            body[off + 12],
            body[off + 13],
            body[off + 14],
            body[off + 15],
        ]);
        let dw_block_start = u32::from_le_bytes([
            body[off + 16],
            body[off + 17],
            body[off + 18],
            body[off + 19],
        ]);
        let dw_sample_offset = u32::from_le_bytes([
            body[off + 20],
            body[off + 21],
            body[off + 22],
            body[off + 23],
        ]);
        let fcc_str = if fcc_chunk.iter().all(|&b| (0x20..=0x7E).contains(&b)) {
            String::from_utf8_lossy(fcc_chunk).to_string()
        } else {
            format!(
                "0x{:02X}{:02X}{:02X}{:02X}",
                fcc_chunk[0], fcc_chunk[1], fcc_chunk[2], fcc_chunk[3]
            )
        };
        out.push((
            format!("wav:cue.{dw_name}.position"),
            dw_position.to_string(),
        ));
        out.push((format!("wav:cue.{dw_name}.fcc_chunk"), fcc_str));
        out.push((
            format!("wav:cue.{dw_name}.chunk_start"),
            dw_chunk_start.to_string(),
        ));
        out.push((
            format!("wav:cue.{dw_name}.block_start"),
            dw_block_start.to_string(),
        ));
        out.push((
            format!("wav:cue.{dw_name}.sample_offset"),
            dw_sample_offset.to_string(),
        ));
    }
}

/// Parse a `plst` (Playlist) chunk body and emit `wav:plst.count` plus
/// per-segment `wav:plst.<n>.cue_id` / `.length` / `.loops` keys. Layout
/// per `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "Playlist
/// Chunk":
///
/// ```text
/// <plst-ck> -> plst( <dwSegments:DWORD> <play-segment>... )
/// <play-segment> -> struct {
///     DWORD dwName;    // cue-point id (must match a <cue-ck> entry)
///     DWORD dwLength;  // section length in samples
///     DWORD dwLoops;   // play count
/// }
/// ```
///
/// The segment index `<n>` is the zero-based position in the playlist,
/// NOT `dwName` — unlike `cue ` / `smpl`-loops, multiple playlist
/// entries can reference the same cue point (a cue replayed twice =
/// two segments with identical `dwName`), so keying on the cue id
/// would collide. A `dwSegments` count exceeding what the body
/// actually carries is clamped to the records that fit (defensive
/// against writers that lie about the count); a body shorter than the
/// 4-byte segment-count header is treated as opaque and skipped.
fn parse_plst_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    if buf.len() < 4 {
        return;
    }
    let count_claimed = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let body = &buf[4..];
    const REC_LEN: usize = 12;
    let count_actual = (body.len() / REC_LEN) as u32;
    let count = count_claimed.min(count_actual);
    out.push(("wav:plst.count".to_string(), count.to_string()));
    for i in 0..count as usize {
        let off = i * REC_LEN;
        let dw_name = u32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]);
        let dw_length =
            u32::from_le_bytes([body[off + 4], body[off + 5], body[off + 6], body[off + 7]]);
        let dw_loops =
            u32::from_le_bytes([body[off + 8], body[off + 9], body[off + 10], body[off + 11]]);
        out.push((format!("wav:plst.{i}.cue_id"), dw_name.to_string()));
        out.push((format!("wav:plst.{i}.length"), dw_length.to_string()));
        out.push((format!("wav:plst.{i}.loops"), dw_loops.to_string()));
    }
}

/// A single cue-point record from a `cue ` chunk.
///
/// Layout per `docs/container/riff/metadata/microsoft-riffmci.pdf` §3
/// "Cue-Points Chunk" — a fixed 24-byte struct:
///
/// ```text
/// <cue-point> -> struct {
///     DWORD  dwName;        // unique id, referenced by plst/adtl
///     DWORD  dwPosition;    // sample position in the play order
///     FOURCC fccChunk;      // 'data' or 'slnt' (for wavl LIST forms)
///     DWORD  dwChunkStart;  // byte offset of fccChunk within wavl LIST
///     DWORD  dwBlockStart;  // byte offset of enclosing block
///     DWORD  dwSampleOffset;// sample offset within block
/// }
/// ```
///
/// For the common single-`data`-chunk WAVE file, `fcc_chunk` is
/// `*b"data"`, `chunk_start` and `block_start` are `0`, and
/// `sample_offset` carries the sample position of the cue point
/// relative to the start of the `data` chunk (RIFF MCI §3 "Examples of
/// File Position Values", row "Within PCM data").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CuePoint {
    /// `dwName` — unique identifier; `plst`/`adtl` records reference a
    /// cue point by this value.
    pub name: u32,
    /// `dwPosition` — sequential sample number within the play order.
    pub position: u32,
    /// `fccChunk` — the chunk ID (`*b"data"` or `*b"slnt"`) that
    /// contains the cue point.
    pub fcc_chunk: [u8; 4],
    /// `dwChunkStart` — byte offset of `fcc_chunk`'s start relative to
    /// the data section of the enclosing `wavl` LIST (`0` for a
    /// single-`data` file).
    pub chunk_start: u32,
    /// `dwBlockStart` — byte offset of the start of the block
    /// containing the position (`0` for a single-`data` file).
    pub block_start: u32,
    /// `dwSampleOffset` — sample offset of the cue point relative to
    /// the start of the block.
    pub sample_offset: u32,
}

impl CuePoint {
    /// On-wire size of a single cue-point record (RIFF MCI §3).
    pub const REC_LEN: usize = 24;

    /// Build a cue point for the common single-`data`-chunk WAVE file:
    /// `fcc_chunk = b"data"`, `chunk_start = block_start = 0`, and the
    /// supplied `sample_offset` (which equals `position` in that
    /// layout).
    pub fn at_sample(name: u32, sample: u32) -> Self {
        CuePoint {
            name,
            position: sample,
            fcc_chunk: *b"data",
            chunk_start: 0,
            block_start: 0,
            sample_offset: sample,
        }
    }

    /// Parse one 24-byte cue-point record. Returns `None` when fewer
    /// than 24 bytes are available.
    pub fn parse(buf: &[u8]) -> Option<CuePoint> {
        if buf.len() < Self::REC_LEN {
            return None;
        }
        let r = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        Some(CuePoint {
            name: r(0),
            position: r(4),
            fcc_chunk: [buf[8], buf[9], buf[10], buf[11]],
            chunk_start: r(12),
            block_start: r(16),
            sample_offset: r(20),
        })
    }

    /// Serialise the 24-byte cue-point record (exact inverse of
    /// [`Self::parse`]).
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut out = [0u8; 24];
        out[0..4].copy_from_slice(&self.name.to_le_bytes());
        out[4..8].copy_from_slice(&self.position.to_le_bytes());
        out[8..12].copy_from_slice(&self.fcc_chunk);
        out[12..16].copy_from_slice(&self.chunk_start.to_le_bytes());
        out[16..20].copy_from_slice(&self.block_start.to_le_bytes());
        out[20..24].copy_from_slice(&self.sample_offset.to_le_bytes());
        out
    }
}

/// A typed view of a `cue ` (cue-points) chunk — the structured
/// counterpart to the read-only `wav:cue.*` metadata keys.
///
/// `cue ` carries a count-prefixed table of [`CuePoint`] records
/// (RIFF MCI §3 "Cue-Points Chunk"). The body is `4 + N*24` bytes,
/// always even, so the muxer never needs a word-alignment pad byte.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CueChunk {
    /// The cue-point table, in file order.
    pub points: Vec<CuePoint>,
}

impl CueChunk {
    /// Build a `cue ` chunk from a list of cue points.
    pub fn new(points: Vec<CuePoint>) -> Self {
        CueChunk { points }
    }

    /// Parse a `cue ` chunk body (`dwCuePoints` count prefix followed
    /// by the cue-point table). A `dwCuePoints` count exceeding what
    /// the body actually carries is clamped to the records that fit
    /// (defensive against writers that lie about the count). A body
    /// shorter than the 4-byte count header returns `None`.
    pub fn parse(buf: &[u8]) -> Option<CueChunk> {
        if buf.len() < 4 {
            return None;
        }
        let claimed = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let body = &buf[4..];
        let fits = (body.len() / CuePoint::REC_LEN) as u32;
        let count = claimed.min(fits);
        let mut points = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let off = i * CuePoint::REC_LEN;
            if let Some(p) = CuePoint::parse(&body[off..]) {
                points.push(p);
            }
        }
        Some(CueChunk { points })
    }

    /// On-wire body length (`4 + N*24` bytes — always even).
    pub fn body_len(&self) -> usize {
        4 + self.points.len() * CuePoint::REC_LEN
    }

    /// Serialise the `cue ` chunk body (count prefix + cue-point
    /// table). Exact inverse of [`Self::parse`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.body_len());
        out.extend_from_slice(&(self.points.len() as u32).to_le_bytes());
        for p in &self.points {
            out.extend_from_slice(&p.to_bytes());
        }
        out
    }
}

/// A single play-segment record from a `plst` (playlist) chunk.
///
/// Layout per `docs/container/riff/metadata/microsoft-riffmci.pdf` §3
/// "Playlist Chunk" — a fixed 12-byte struct:
///
/// ```text
/// <play-segment> -> struct {
///     DWORD dwName;   // cue-point id (must match a <cue-ck> entry)
///     DWORD dwLength; // section length in samples
///     DWORD dwLoops;  // play count
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaylistSegment {
    /// `dwName` — the cue-point id this segment plays from; must match
    /// a [`CuePoint::name`] in the file's `cue ` table.
    pub cue_id: u32,
    /// `dwLength` — length of the section in samples.
    pub length: u32,
    /// `dwLoops` — number of times to play the section.
    pub loops: u32,
}

impl PlaylistSegment {
    /// On-wire size of a single play-segment record (RIFF MCI §3).
    pub const REC_LEN: usize = 12;

    /// Parse one 12-byte play-segment record. Returns `None` when
    /// fewer than 12 bytes are available.
    pub fn parse(buf: &[u8]) -> Option<PlaylistSegment> {
        if buf.len() < Self::REC_LEN {
            return None;
        }
        let r = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        Some(PlaylistSegment {
            cue_id: r(0),
            length: r(4),
            loops: r(8),
        })
    }

    /// Serialise the 12-byte play-segment record (exact inverse of
    /// [`Self::parse`]).
    pub fn to_bytes(&self) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(&self.cue_id.to_le_bytes());
        out[4..8].copy_from_slice(&self.length.to_le_bytes());
        out[8..12].copy_from_slice(&self.loops.to_le_bytes());
        out
    }
}

/// A typed view of a `plst` (playlist) chunk — the structured
/// counterpart to the read-only `wav:plst.*` metadata keys.
///
/// `plst` carries a count-prefixed table of [`PlaylistSegment`]
/// records (RIFF MCI §3 "Playlist Chunk"). The body is `4 + N*12`
/// bytes, always even.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlaylistChunk {
    /// The play-segment table, in file order.
    pub segments: Vec<PlaylistSegment>,
}

impl PlaylistChunk {
    /// Build a `plst` chunk from a list of play segments.
    pub fn new(segments: Vec<PlaylistSegment>) -> Self {
        PlaylistChunk { segments }
    }

    /// Parse a `plst` chunk body. A `dwSegments` count exceeding what
    /// the body carries is clamped; a body shorter than the 4-byte
    /// count header returns `None`.
    pub fn parse(buf: &[u8]) -> Option<PlaylistChunk> {
        if buf.len() < 4 {
            return None;
        }
        let claimed = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let body = &buf[4..];
        let fits = (body.len() / PlaylistSegment::REC_LEN) as u32;
        let count = claimed.min(fits);
        let mut segments = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let off = i * PlaylistSegment::REC_LEN;
            if let Some(s) = PlaylistSegment::parse(&body[off..]) {
                segments.push(s);
            }
        }
        Some(PlaylistChunk { segments })
    }

    /// On-wire body length (`4 + N*12` bytes — always even).
    pub fn body_len(&self) -> usize {
        4 + self.segments.len() * PlaylistSegment::REC_LEN
    }

    /// Serialise the `plst` chunk body. Exact inverse of
    /// [`Self::parse`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.body_len());
        out.extend_from_slice(&(self.segments.len() as u32).to_le_bytes());
        for s in &self.segments {
            out.extend_from_slice(&s.to_bytes());
        }
        out
    }
}

/// One entry of a `LIST adtl` (Associated Data List) chunk.
///
/// Layout per `docs/container/riff/metadata/microsoft-riffmci.pdf` §3
/// "Associated Data Chunk". Each variant attaches text or metadata to
/// a cue point identified by `dwName` (which must match a
/// [`CuePoint::name`] in the file's `cue ` table):
///
/// - `labl( <dwName:DWORD> <data:ZSTR> )` — a label/title.
/// - `note( <dwName:DWORD> <data:ZSTR> )` — comment text.
/// - `ltxt( <dwName> <dwSampleLength> <dwPurpose> <wCountry> <wLanguage>
///   <wDialect> <wCodePage> <data:BYTE>... )` — text spanning a
///   `dwSampleLength`-sample segment, qualified by the CSET-style
///   country / language / dialect / code-page fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdtlEntry {
    /// `labl` — a label/title for a cue point.
    Label {
        /// `dwName` — the cue-point id this label is attached to.
        name: u32,
        /// NUL-terminated label text (stored without the terminator).
        text: String,
    },
    /// `note` — comment text for a cue point.
    Note {
        /// `dwName` — the cue-point id this note is attached to.
        name: u32,
        /// NUL-terminated comment text (stored without the terminator).
        text: String,
    },
    /// `ltxt` — text associated with a `sample_length`-sample segment.
    LabeledText {
        /// `dwName` — the cue-point id this segment is anchored to.
        name: u32,
        /// `dwSampleLength` — number of samples the segment spans.
        sample_length: u32,
        /// `dwPurpose` — FOURCC purpose code (e.g. `*b"scrp"` for
        /// script text, `*b"capt"` for closed-caption text).
        purpose: [u8; 4],
        /// `wCountry` — country code (RIFF MCI Chapter 2 table; `0` =
        /// default).
        country: u16,
        /// `wLanguage` — language code (RIFF MCI Chapter 2 table).
        language: u16,
        /// `wDialect` — dialect code (RIFF MCI Chapter 2 table).
        dialect: u16,
        /// `wCodePage` — code page for the text payload.
        code_page: u16,
        /// The text payload (stored without any trailing NUL).
        text: String,
    },
}

impl AdtlEntry {
    /// The sub-chunk FOURCC for this entry (`labl` / `note` / `ltxt`).
    fn fourcc(&self) -> &'static [u8; 4] {
        match self {
            AdtlEntry::Label { .. } => b"labl",
            AdtlEntry::Note { .. } => b"note",
            AdtlEntry::LabeledText { .. } => b"ltxt",
        }
    }

    /// On-wire body length (excluding the 8-byte sub-chunk header and
    /// any word-alignment pad).
    fn body_len(&self) -> usize {
        match self {
            AdtlEntry::Label { text, .. } | AdtlEntry::Note { text, .. } => 4 + text.len() + 1,
            AdtlEntry::LabeledText { text, .. } => 20 + text.len(),
        }
    }

    /// Serialise this entry's sub-chunk body (without the 8-byte
    /// header). `labl`/`note` bodies are NUL-terminated; `ltxt` is the
    /// 20-byte fixed header followed by the raw (un-terminated) text.
    fn body_bytes(&self) -> Vec<u8> {
        match self {
            AdtlEntry::Label { name, text } | AdtlEntry::Note { name, text } => {
                let mut out = Vec::with_capacity(4 + text.len() + 1);
                out.extend_from_slice(&name.to_le_bytes());
                out.extend_from_slice(text.as_bytes());
                out.push(0); // ZSTR terminator
                out
            }
            AdtlEntry::LabeledText {
                name,
                sample_length,
                purpose,
                country,
                language,
                dialect,
                code_page,
                text,
            } => {
                let mut out = Vec::with_capacity(20 + text.len());
                out.extend_from_slice(&name.to_le_bytes());
                out.extend_from_slice(&sample_length.to_le_bytes());
                out.extend_from_slice(purpose);
                out.extend_from_slice(&country.to_le_bytes());
                out.extend_from_slice(&language.to_le_bytes());
                out.extend_from_slice(&dialect.to_le_bytes());
                out.extend_from_slice(&code_page.to_le_bytes());
                out.extend_from_slice(text.as_bytes());
                out
            }
        }
    }

    /// Parse one `labl` / `note` / `ltxt` sub-chunk body (the bytes
    /// after the 8-byte sub-chunk header). Returns `None` for an
    /// unrecognised id or a truncated body.
    fn parse(id: &[u8; 4], body: &[u8]) -> Option<AdtlEntry> {
        match id {
            b"labl" | b"note" if body.len() >= 4 => {
                let name = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);
                let raw = &body[4..];
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                let text = String::from_utf8_lossy(&raw[..end]).into_owned();
                if id == b"labl" {
                    Some(AdtlEntry::Label { name, text })
                } else {
                    Some(AdtlEntry::Note { name, text })
                }
            }
            b"ltxt" if body.len() >= 20 => {
                let r32 =
                    |o: usize| u32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]);
                let r16 = |o: usize| u16::from_le_bytes([body[o], body[o + 1]]);
                let raw = &body[20..];
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                Some(AdtlEntry::LabeledText {
                    name: r32(0),
                    sample_length: r32(4),
                    purpose: [body[8], body[9], body[10], body[11]],
                    country: r16(12),
                    language: r16(14),
                    dialect: r16(16),
                    code_page: r16(18),
                    text: String::from_utf8_lossy(&raw[..end]).into_owned(),
                })
            }
            _ => None,
        }
    }
}

/// A typed view of a `LIST adtl` (Associated Data List) chunk — the
/// structured counterpart to the read-only `wav:adtl.*` metadata keys.
///
/// The list groups [`AdtlEntry`] records (`labl`/`note`/`ltxt`) that
/// annotate the cue points declared by a sibling `cue ` chunk (RIFF
/// MCI §3 "Associated Data Chunk"). The `file` sub-chunk (embedded
/// media) is parsed for its `dwName`/`dwMedType` accounting through
/// the metadata path but is not represented as a typed [`AdtlEntry`]
/// variant (its payload is opaque to this layer).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdtlChunk {
    /// The associated-data entries, in file order.
    pub entries: Vec<AdtlEntry>,
}

impl AdtlChunk {
    /// Build an `adtl` list from a vector of entries.
    pub fn new(entries: Vec<AdtlEntry>) -> Self {
        AdtlChunk { entries }
    }

    /// Parse a `LIST adtl` body (the bytes *after* the 4-byte `adtl`
    /// list type). Unrecognised / truncated sub-chunks are skipped.
    pub fn parse(buf: &[u8]) -> AdtlChunk {
        let mut entries = Vec::new();
        let mut i = 0usize;
        while i + 8 <= buf.len() {
            let id: [u8; 4] = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
            let size =
                u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
            i += 8;
            if i + size > buf.len() {
                break;
            }
            if let Some(entry) = AdtlEntry::parse(&id, &buf[i..i + size]) {
                entries.push(entry);
            }
            i += size;
            if size % 2 == 1 {
                i += 1; // word-alignment pad
            }
        }
        AdtlChunk { entries }
    }

    /// On-wire body length of the whole `LIST adtl` chunk *body* — the
    /// 4-byte `adtl` list type plus, for each sub-chunk, its 8-byte
    /// header, body bytes and word-alignment pad. This is the value
    /// written into the enclosing `LIST` chunk's size field.
    pub fn list_body_len(&self) -> usize {
        let mut len = 4; // "adtl" list type
        for e in &self.entries {
            let body = e.body_len();
            len += 8 + body + (body % 2);
        }
        len
    }

    /// Serialise the whole `LIST adtl` chunk *body*: the `adtl` list
    /// type followed by each entry's sub-chunk (`id` + size + body +
    /// word-alignment pad). The caller prepends the `LIST` id and the
    /// total size. Exact inverse of [`Self::parse`] modulo the list
    /// type prefix.
    pub fn to_list_body(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.list_body_len());
        out.extend_from_slice(b"adtl");
        for e in &self.entries {
            let body = e.body_bytes();
            out.extend_from_slice(e.fourcc());
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(&body);
            if body.len() % 2 == 1 {
                out.push(0); // word-alignment pad
            }
        }
        out
    }
}

/// Parse a `smpl` (Sampler) chunk body and emit `wav:smpl.*` metadata
/// keys. Layout per the RIFF MCI / `mmreg.h`-era Sampler structure as
/// catalogued in
/// `docs/container/riff/metadata/exiftool-riff-tags.html` § "RIFF
/// Sampler Tags" and summarised in
/// `docs/container/riff/metadata/README.md` § "Sampler / Instrument
/// chunks":
///
/// ```text
/// <smpl-ck> -> smpl( <fixed:36 bytes> <loops:N × 24 bytes> <sampler-data> )
/// <fixed> -> struct {
///     DWORD dwManufacturer;
///     DWORD dwProduct;
///     DWORD dwSamplePeriod;       // nanoseconds per sample
///     DWORD dwMIDIUnityNote;      // 0..=127
///     DWORD dwMIDIPitchFraction;  // fractional offset, /0xFFFFFFFF
///     DWORD dwSMPTEFormat;        // 0/24/25/29/30 fps
///     DWORD dwSMPTEOffset;        // packed HH MM SS FF (MSB → LSB)
///     DWORD cSampleLoops;
///     DWORD cbSamplerData;        // bytes of trailing sampler-specific data
/// }
/// <loop> -> struct {
///     DWORD dwCuePointID;
///     DWORD dwType;               // 0=forward, 1=ping-pong, 2=reverse
///     DWORD dwStart;              // start sample offset
///     DWORD dwEnd;                // end sample offset
///     DWORD dwFraction;           // /0xFFFFFFFF fractional sample
///     DWORD dwPlayCount;          // 0 == infinite
/// }
/// ```
///
/// A `cSampleLoops` count exceeding what the chunk body actually
/// carries is clamped to the records that fit (defensive against
/// writers that lie about the count); a body shorter than the 36-byte
/// fixed header is treated as opaque and skipped.
fn parse_smpl_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    const FIXED_LEN: usize = 36;
    const LOOP_LEN: usize = 24;
    if buf.len() < FIXED_LEN {
        return;
    }
    let r = |off: usize| -> u32 {
        u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    };
    let manufacturer = r(0);
    let product = r(4);
    let sample_period = r(8);
    let midi_unity_note = r(12);
    let midi_pitch_fraction = r(16);
    let smpte_format = r(20);
    let smpte_offset = r(24);
    let num_loops_claimed = r(28);
    let sampler_data_len = r(32);

    out.push((
        "wav:smpl.manufacturer".to_string(),
        manufacturer.to_string(),
    ));
    out.push(("wav:smpl.product".to_string(), product.to_string()));
    out.push((
        "wav:smpl.sample_period".to_string(),
        sample_period.to_string(),
    ));
    out.push((
        "wav:smpl.midi_unity_note".to_string(),
        midi_unity_note.to_string(),
    ));
    out.push((
        "wav:smpl.midi_pitch_fraction".to_string(),
        midi_pitch_fraction.to_string(),
    ));
    out.push((
        "wav:smpl.smpte_format".to_string(),
        smpte_format.to_string(),
    ));
    // SMPTE offset packs HH MM SS FF in the high-to-low bytes of the
    // DWORD. Render the canonical HH:MM:SS:FF form alongside the raw
    // value for callers that prefer the on-wire integer.
    let smpte_hh = (smpte_offset >> 24) & 0xFF;
    let smpte_mm = (smpte_offset >> 16) & 0xFF;
    let smpte_ss = (smpte_offset >> 8) & 0xFF;
    let smpte_ff = smpte_offset & 0xFF;
    out.push((
        "wav:smpl.smpte_offset".to_string(),
        format!("{smpte_hh:02}:{smpte_mm:02}:{smpte_ss:02}:{smpte_ff:02}"),
    ));
    out.push((
        "wav:smpl.sampler_data_len".to_string(),
        sampler_data_len.to_string(),
    ));

    let body = &buf[FIXED_LEN..];
    let num_loops_fits = (body.len() / LOOP_LEN) as u32;
    let num_loops = num_loops_claimed.min(num_loops_fits);
    out.push((
        "wav:smpl.num_sample_loops".to_string(),
        num_loops.to_string(),
    ));
    for i in 0..num_loops as usize {
        let off = i * LOOP_LEN;
        let loop_field = |word: usize| -> u32 {
            let p = off + word * 4;
            u32::from_le_bytes([body[p], body[p + 1], body[p + 2], body[p + 3]])
        };
        let cue_point_id = loop_field(0);
        let loop_type = loop_field(1);
        let start = loop_field(2);
        let end = loop_field(3);
        let fraction = loop_field(4);
        let play_count = loop_field(5);
        out.push((
            format!("wav:smpl.loop.{i}.cue_point_id"),
            cue_point_id.to_string(),
        ));
        out.push((format!("wav:smpl.loop.{i}.type"), loop_type.to_string()));
        out.push((format!("wav:smpl.loop.{i}.start"), start.to_string()));
        out.push((format!("wav:smpl.loop.{i}.end"), end.to_string()));
        out.push((format!("wav:smpl.loop.{i}.fraction"), fraction.to_string()));
        out.push((
            format!("wav:smpl.loop.{i}.play_count"),
            play_count.to_string(),
        ));
    }
}

/// Parse an `inst` (Instrument) chunk body and emit `wav:inst.*`
/// metadata keys. Layout per
/// `docs/container/riff/metadata/exiftool-riff-tags.html` § "RIFF
/// Instrument Tags":
///
/// ```text
/// <inst-ck> -> inst( <UnshiftedNote:i8> <FineTune:i8> <Gain:i8>
///                    <LowNote:u8> <HighNote:u8>
///                    <LowVelocity:u8> <HighVelocity:u8> )
/// ```
///
/// `UnshiftedNote`, `LowNote`, `HighNote` are MIDI note numbers
/// (0..=127); `FineTune` is cents and `Gain` is dB, both signed
/// 8-bit. Velocity range is 1..=127 (unsigned).
///
/// A body shorter than the 7-byte fixed struct is treated as opaque
/// and skipped.
fn parse_inst_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    const FIXED_LEN: usize = 7;
    if buf.len() < FIXED_LEN {
        return;
    }
    let unshifted_note = buf[0];
    let fine_tune = buf[1] as i8;
    let gain = buf[2] as i8;
    let low_note = buf[3];
    let high_note = buf[4];
    let low_velocity = buf[5];
    let high_velocity = buf[6];
    out.push((
        "wav:inst.unshifted_note".to_string(),
        unshifted_note.to_string(),
    ));
    out.push(("wav:inst.fine_tune".to_string(), fine_tune.to_string()));
    out.push(("wav:inst.gain".to_string(), gain.to_string()));
    out.push(("wav:inst.low_note".to_string(), low_note.to_string()));
    out.push(("wav:inst.high_note".to_string(), high_note.to_string()));
    out.push((
        "wav:inst.low_velocity".to_string(),
        low_velocity.to_string(),
    ));
    out.push((
        "wav:inst.high_velocity".to_string(),
        high_velocity.to_string(),
    ));
}

/// `AcidChunk::flags` bit 0 — the clip is a one-shot (played once,
/// not tempo-stretched as a loop).
pub const ACID_FLAG_ONE_SHOT: u32 = 1 << 0;
/// `AcidChunk::flags` bit 1 — `root_note` carries a meaningful value.
pub const ACID_FLAG_ROOT_NOTE_SET: u32 = 1 << 1;
/// `AcidChunk::flags` bit 2 — time-stretch enabled.
pub const ACID_FLAG_STRETCH: u32 = 1 << 2;
/// `AcidChunk::flags` bit 3 — disk-based (streamed) rather than
/// RAM-resident.
pub const ACID_FLAG_DISK_BASED: u32 = 1 << 3;
/// `AcidChunk::flags` bit 4 — high-octave root-note interpretation.
pub const ACID_FLAG_HIGH_OCTAVE: u32 = 1 << 4;

/// Typed view of the Acidizer `acid` chunk body (loop/tempo metadata
/// written by loop-authoring tools). Field offsets and flag-bit
/// semantics per
/// `docs/container/riff/metadata/exiftool-riff-tags.html` § "RIFF
/// Acidizer Tags" (byte-indexed table: flags at 0, root note at 4,
/// beats at 12, meter at 16, tempo at 20):
///
/// ```text
/// <acid-ck> -> acid( <Flags:u32> <RootNote:u16> <Reserved:[u8;6]>
///                    <Beats:u32> <Meter:u32> <Tempo:f32> )   // 24 bytes
/// ```
///
/// All integer fields are little-endian. The staged reference
/// enumerates the field *offsets*; the widths follow from the offset
/// deltas (flags 0..4, beats 12..16, meter 16..20, tempo 20..24).
/// Within the 8-byte span between the root-note offset (4) and the
/// beats offset (12) only the leading 16 bits are enumerated (root
/// note, value range 48..=71 per the table) — the remaining 6 bytes
/// are not described, so they are carried verbatim in [`Self::reserved`]
/// and round-trip losslessly. `tempo` is the 32-bit field at offset
/// 20, interpreted as an IEEE-754 little-endian beats-per-minute
/// value (the only 32-bit reading under which musically plausible
/// tempos are representable with fractional precision).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcidChunk {
    /// Bit-field at offset 0 — see the `ACID_FLAG_*` constants.
    pub flags: u32,
    /// Root note at offset 4. 48 = C up to 71 = High B per the staged
    /// table; meaningful only when [`Self::root_note_set`] is true.
    pub root_note: u16,
    /// Bytes 6..12 — not enumerated by the staged reference; preserved
    /// verbatim so a read→write pass is byte-lossless.
    pub reserved: [u8; 6],
    /// Number of beats in the clip (offset 12).
    pub num_beats: u32,
    /// Meter field at offset 16 (single 32-bit field in the staged
    /// table; carried raw).
    pub meter: u32,
    /// Tempo in beats per minute (offset 20).
    pub tempo: f32,
}

impl AcidChunk {
    /// Fixed body length of the `acid` chunk in bytes.
    pub const BODY_LEN: usize = 24;

    /// Bit 0 — one-shot clip.
    pub fn one_shot(&self) -> bool {
        self.flags & ACID_FLAG_ONE_SHOT != 0
    }

    /// Bit 1 — `root_note` carries a meaningful value.
    pub fn root_note_set(&self) -> bool {
        self.flags & ACID_FLAG_ROOT_NOTE_SET != 0
    }

    /// Bit 2 — time-stretch enabled.
    pub fn stretch(&self) -> bool {
        self.flags & ACID_FLAG_STRETCH != 0
    }

    /// Bit 3 — disk-based (streamed).
    pub fn disk_based(&self) -> bool {
        self.flags & ACID_FLAG_DISK_BASED != 0
    }

    /// Bit 4 — high-octave root-note interpretation.
    pub fn high_octave(&self) -> bool {
        self.flags & ACID_FLAG_HIGH_OCTAVE != 0
    }

    /// Note name for [`Self::root_note`] per the staged value table
    /// (48 = C … 59 = B, 60 = High C … 71 = High B). `None` outside
    /// the enumerated 48..=71 range.
    pub fn root_note_name(&self) -> Option<&'static str> {
        const NAMES: [&str; 24] = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "High C", "High C#",
            "High D", "High D#", "High E", "High F", "High F#", "High G", "High G#", "High A",
            "High A#", "High B",
        ];
        NAMES.get(self.root_note.wrapping_sub(48) as usize).copied()
    }

    /// Decode an `acid` chunk body. Returns `None` when the body is
    /// shorter than the 24-byte fixed struct (treated as opaque, same
    /// policy as the other fixed-layout metadata chunks). Trailing
    /// bytes past offset 24 are tolerated and ignored.
    pub fn parse(buf: &[u8]) -> Option<AcidChunk> {
        if buf.len() < Self::BODY_LEN {
            return None;
        }
        let r32 =
            |o: usize| -> u32 { u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]) };
        let mut reserved = [0u8; 6];
        reserved.copy_from_slice(&buf[6..12]);
        Some(AcidChunk {
            flags: r32(0),
            root_note: u16::from_le_bytes([buf[4], buf[5]]),
            reserved,
            num_beats: r32(12),
            meter: r32(16),
            tempo: f32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]),
        })
    }

    /// Serialize the 24-byte `acid` chunk body (little-endian, layout
    /// per the struct-level documentation).
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut out = [0u8; 24];
        out[0..4].copy_from_slice(&self.flags.to_le_bytes());
        out[4..6].copy_from_slice(&self.root_note.to_le_bytes());
        out[6..12].copy_from_slice(&self.reserved);
        out[12..16].copy_from_slice(&self.num_beats.to_le_bytes());
        out[16..20].copy_from_slice(&self.meter.to_le_bytes());
        out[20..24].copy_from_slice(&self.tempo.to_le_bytes());
        out
    }
}

/// Parse an `acid` chunk body, surface `wav:acid.*` metadata keys and
/// return the typed view for the demuxer's [`WavDemuxer::acid`]
/// accessor. Keys:
///
/// - `wav:acid.flags` — bit-field as `0xXXXXXXXX`.
/// - `wav:acid.one_shot` / `.root_note_set` / `.stretch` /
///   `.disk_based` / `.high_octave` — each documented flag bit as
///   `0` / `1`.
/// - `wav:acid.root_note` — raw value, plus `wav:acid.root_note_name`
///   when the value falls in the enumerated 48..=71 table.
/// - `wav:acid.num_beats`, `wav:acid.meter`, `wav:acid.tempo`.
/// - `wav:acid.reserved` — bytes 6..12 as hex, only when nonzero.
/// - `wav:acid.body_len` — only when the body exceeds the 24-byte
///   fixed struct (extension bytes riding along).
fn parse_acid_chunk(buf: &[u8], out: &mut Vec<(String, String)>) -> Option<AcidChunk> {
    let acid = AcidChunk::parse(buf)?;
    out.push((
        "wav:acid.flags".to_string(),
        format!("0x{:08X}", acid.flags),
    ));
    out.push((
        "wav:acid.one_shot".to_string(),
        (acid.one_shot() as u8).to_string(),
    ));
    out.push((
        "wav:acid.root_note_set".to_string(),
        (acid.root_note_set() as u8).to_string(),
    ));
    out.push((
        "wav:acid.stretch".to_string(),
        (acid.stretch() as u8).to_string(),
    ));
    out.push((
        "wav:acid.disk_based".to_string(),
        (acid.disk_based() as u8).to_string(),
    ));
    out.push((
        "wav:acid.high_octave".to_string(),
        (acid.high_octave() as u8).to_string(),
    ));
    out.push(("wav:acid.root_note".to_string(), acid.root_note.to_string()));
    if let Some(name) = acid.root_note_name() {
        out.push(("wav:acid.root_note_name".to_string(), name.to_string()));
    }
    out.push(("wav:acid.num_beats".to_string(), acid.num_beats.to_string()));
    out.push(("wav:acid.meter".to_string(), acid.meter.to_string()));
    out.push(("wav:acid.tempo".to_string(), acid.tempo.to_string()));
    if acid.reserved.iter().any(|&b| b != 0) {
        let hex: String = acid.reserved.iter().map(|b| format!("{b:02X}")).collect();
        out.push(("wav:acid.reserved".to_string(), hex));
    }
    if buf.len() > AcidChunk::BODY_LEN {
        out.push(("wav:acid.body_len".to_string(), buf.len().to_string()));
    }
    Some(acid)
}

/// One `audioID` record inside a BW64/ADM `chna` chunk — 40 bytes,
/// fixed layout per ITU-R BS.2088-2 §8.1
/// (`docs/container/riff/metadata/bs2088-chna-chunk-layout.md` §1.2):
///
/// ```text
/// struct audioID {                 // 40 bytes
///     WORD     trackIndex;         // off 0  — 1-based index into <data>; 0 = unused
///     CHAR[12] UID;                // off 2  — audioTrackUID  "ATU_xxxxxxxx"
///     CHAR[14] trackRef;           // off 14 — audioTrackFormatID "AT_xxxxxxxx_xx"
///                                  //          (or audioChannelFormat "AC_xxxxxxxx_00")
///     CHAR[11] packRef;            // off 28 — audioPackFormatID "AP_xxxxxxxx" (or 11 NULs)
///     CHAR     pad;                // off 39 — padding, makes the record even-sized
/// }
/// ```
///
/// The character-array fields are fixed-width, **not** NUL-terminated
/// ASCII (§1.1); unused records (`track_index == 0`) zero every field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioId {
    /// `trackIndex` (off 0) — 1-based index of the track in the `data`
    /// chunk interleave. `0` marks an unused (spare) record.
    pub track_index: u16,
    /// `UID` (off 2) — 12-byte `audioTrackUID`, format `ATU_xxxxxxxx`.
    /// Fixed-width raw bytes (all-zero for an unused record).
    pub uid: [u8; 12],
    /// `trackRef` (off 14) — 14-byte `audioTrackFormatID` reference
    /// (`AT_xxxxxxxx_xx`), or an `audioChannelFormat` ref
    /// (`AC_xxxxxxxx_00`) for linear-PCM essence.
    pub track_ref: [u8; 14],
    /// `packRef` (off 28) — 11-byte `audioPackFormatID` reference
    /// (`AP_xxxxxxxx`). All 11 bytes NUL when no pack is required.
    pub pack_ref: [u8; 11],
    /// `pad` (off 39) — single trailing padding byte (§1.2 uses `\0`),
    /// carried verbatim so a read→write pass is byte-lossless.
    pub pad: u8,
}

/// Which ADM identifier an `audioID` reference field names, derived from
/// the fixed ASCII prefix the BS.2088-2 §8.2 ID formats use:
///
/// - `ATU_xxxxxxxx`    → [`AdmRefKind::TrackUid`]      (`audioTrackUID`)
/// - `AT_xxxxxxxx_xx`  → [`AdmRefKind::TrackFormat`]   (`audioTrackFormatID`)
/// - `AC_xxxxxxxx_xx`  → [`AdmRefKind::ChannelFormat`] (`audioChannelFormat`,
///   used in `trackRef` for linear-PCM essence — §1.2)
/// - `AP_xxxxxxxx`     → [`AdmRefKind::PackFormat`]    (`audioPackFormatID`)
///
/// `Unknown` is returned for any field that doesn't start with one of the
/// four documented prefixes (e.g. an all-NUL spare field, or a malformed
/// writer's output).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmRefKind {
    /// `ATU_` — an `audioTrackUID`.
    TrackUid,
    /// `AT_` — an `audioTrackFormatID`.
    TrackFormat,
    /// `AC_` — an `audioChannelFormat` reference (linear-PCM essence path).
    ChannelFormat,
    /// `AP_` — an `audioPackFormatID`.
    PackFormat,
    /// None of the four documented prefixes matched.
    Unknown,
}

impl AdmRefKind {
    /// Classify a fixed-width `audioID` reference field by its leading
    /// ASCII prefix. Order matters: `ATU_` must be tested before `AT_`
    /// since the latter is a strict prefix of the former.
    fn classify(field: &[u8]) -> AdmRefKind {
        if field.starts_with(b"ATU_") {
            AdmRefKind::TrackUid
        } else if field.starts_with(b"AT_") {
            AdmRefKind::TrackFormat
        } else if field.starts_with(b"AC_") {
            AdmRefKind::ChannelFormat
        } else if field.starts_with(b"AP_") {
            AdmRefKind::PackFormat
        } else {
            AdmRefKind::Unknown
        }
    }

    /// Short lowercase tag used in surfaced metadata values.
    fn as_str(self) -> &'static str {
        match self {
            AdmRefKind::TrackUid => "audioTrackUID",
            AdmRefKind::TrackFormat => "audioTrackFormatID",
            AdmRefKind::ChannelFormat => "audioChannelFormat",
            AdmRefKind::PackFormat => "audioPackFormatID",
            AdmRefKind::Unknown => "unknown",
        }
    }
}

/// Whether an ADM ID resolves to a **common** definition (built into
/// ITU-R BS.2094 and not required in the file's XML) or a **custom**
/// definition (carried in the file's `<axml>`/`<bxml>`/`<sxml>` chunk),
/// per BS.2088-2 §8.1 (`bs2088-chna-chunk-layout.md` §3).
///
/// The discriminator is the **last four hex digits** of the ID value:
/// `≤ 0x0FFF` → common, `≥ 0x1000` → custom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefinitionScope {
    /// Trailing four hex digits `≤ 0x0FFF` — a BS.2094 common definition.
    Common,
    /// Trailing four hex digits `≥ 0x1000` — a file-local custom
    /// definition (lives in the XML carrier chunk).
    Custom,
}

impl DefinitionScope {
    /// Lowercase tag used in surfaced metadata values.
    fn as_str(self) -> &'static str {
        match self {
            DefinitionScope::Common => "common",
            DefinitionScope::Custom => "custom",
        }
    }
}

/// Extract the **last four hex digits** of an ADM reference's value, as a
/// `u16`, for the §3 common/custom classification.
///
/// The ID formats all encode an 8-hex-digit value after the prefix
/// (`AT_`**`xxxxxxxx`**`_xx`, `AP_`**`xxxxxxxx`**, …). The classification
/// (`bs2088-chna-chunk-layout.md` §3) keys only on the trailing four hex
/// digits of that 8-digit value — i.e. the lower 16 bits. We locate the
/// 8-hex-digit run that follows the first `_` separator and parse its
/// last four characters. Returns `None` when the field has no parseable
/// 8-hex value (all-NUL, malformed, or an unknown prefix).
fn adm_id_value_low16(field: &[u8]) -> Option<u16> {
    // Trim at the first NUL (fixed-width fields are NUL-padded).
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    let s = &field[..end];
    // Find the first `_`; the 8 hex digits begin immediately after it.
    let us = s.iter().position(|&b| b == b'_')?;
    let hex = s.get(us + 1..us + 1 + 8)?;
    if !hex.iter().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    // The low 16 bits are the trailing four hex digits of the 8-digit run.
    let last4 = &hex[4..8];
    u16::from_str_radix(std::str::from_utf8(last4).ok()?, 16).ok()
}

/// Classify an ADM reference field's definition scope (§3): `Common` when
/// the trailing four hex digits of its value are `≤ 0x0FFF`, `Custom`
/// when `≥ 0x1000`. Returns `None` when no 8-hex value can be parsed.
fn adm_definition_scope(field: &[u8]) -> Option<DefinitionScope> {
    let low = adm_id_value_low16(field)?;
    Some(if low <= 0x0FFF {
        DefinitionScope::Common
    } else {
        DefinitionScope::Custom
    })
}

impl AudioId {
    /// Fixed on-disk size of one `audioID` record in bytes (§1.2).
    pub const SIZE: usize = 40;

    /// `true` when this record is the spare/unused marker
    /// (`track_index == 0`) — readers skip these (§1.3).
    pub fn is_unused(&self) -> bool {
        self.track_index == 0
    }

    /// Classify the `trackRef` field (§1.2): `AT_` =
    /// [`AdmRefKind::TrackFormat`] or, for linear-PCM essence, `AC_` =
    /// [`AdmRefKind::ChannelFormat`].
    pub fn track_ref_kind(&self) -> AdmRefKind {
        AdmRefKind::classify(&self.track_ref)
    }

    /// Classify the `packRef` field (§1.2): `AP_` =
    /// [`AdmRefKind::PackFormat`], or [`AdmRefKind::Unknown`] when the
    /// field is all-NUL (no pack required).
    pub fn pack_ref_kind(&self) -> AdmRefKind {
        AdmRefKind::classify(&self.pack_ref)
    }

    /// Definition scope of the `trackRef` (§3): `Common` (BS.2094) when
    /// the trailing four hex digits of the ID value are `≤ 0x0FFF`,
    /// `Custom` (in the XML carrier) when `≥ 0x1000`. `None` when the
    /// field has no parseable 8-hex value.
    pub fn track_ref_scope(&self) -> Option<DefinitionScope> {
        adm_definition_scope(&self.track_ref)
    }

    /// Definition scope of the `packRef` (§3), same rule as
    /// [`Self::track_ref_scope`]. `None` for an all-NUL pack field.
    pub fn pack_ref_scope(&self) -> Option<DefinitionScope> {
        adm_definition_scope(&self.pack_ref)
    }

    /// Decode one 40-byte `audioID` record. Returns `None` when `buf`
    /// is shorter than [`Self::SIZE`].
    pub fn parse(buf: &[u8]) -> Option<AudioId> {
        if buf.len() < Self::SIZE {
            return None;
        }
        let mut uid = [0u8; 12];
        uid.copy_from_slice(&buf[2..14]);
        let mut track_ref = [0u8; 14];
        track_ref.copy_from_slice(&buf[14..28]);
        let mut pack_ref = [0u8; 11];
        pack_ref.copy_from_slice(&buf[28..39]);
        Some(AudioId {
            track_index: u16::from_le_bytes([buf[0], buf[1]]),
            uid,
            track_ref,
            pack_ref,
            pad: buf[39],
        })
    }

    /// Serialize the 40-byte `audioID` record (little-endian
    /// `track_index`, fixed-width raw char arrays, trailing `pad`).
    pub fn to_bytes(&self) -> [u8; 40] {
        let mut out = [0u8; 40];
        out[0..2].copy_from_slice(&self.track_index.to_le_bytes());
        out[2..14].copy_from_slice(&self.uid);
        out[14..28].copy_from_slice(&self.track_ref);
        out[28..39].copy_from_slice(&self.pack_ref);
        out[39] = self.pad;
        out
    }
}

/// Render a fixed-width `audioID` char-array field as text: ASCII bytes
/// up to the first NUL (or the full width when there is no NUL),
/// dropping any trailing NUL padding. Returns `None` when the field is
/// entirely NUL (e.g. a `pack_ref` with no pack, or an unused record).
fn audio_id_text(field: &[u8]) -> Option<String> {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    if end == 0 {
        return None;
    }
    Some(String::from_utf8_lossy(&field[..end]).into_owned())
}

/// Typed view of the BW64/ADM `chna` (channel allocation) chunk per
/// ITU-R BS.2088-2 §8.1
/// (`docs/container/riff/metadata/bs2088-chna-chunk-layout.md`):
///
/// ```text
/// <chna-ck> -> chna( <numTracks:WORD> <numUIDs:WORD> <audioID[N]> )
/// ```
///
/// where the record count `N = (ckSize - 4) / 40` (§1.1). `N` may
/// exceed `num_uids` because writers over-provision spare records for
/// later in-place editing (§1.3); the spare records have
/// `track_index == 0` and are carried verbatim so a read→write pass is
/// byte-lossless.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChnaChunk {
    /// `numTracks` (off 0) — number of tracks used in the file. A track
    /// carrying multiple ID sets still counts as one (§1.1).
    pub num_tracks: u16,
    /// `numUIDs` (off 2) — number of UIDs used; equals the number of
    /// defined (non-zero) `ID[]` records and may exceed `num_tracks`
    /// (§1.1).
    pub num_uids: u16,
    /// The `audioID` record array (`N` entries, `N >= num_uids`).
    /// Includes any spare (`track_index == 0`) records so the chunk
    /// round-trips byte-for-byte.
    pub ids: Vec<AudioId>,
}

impl ChnaChunk {
    /// Fixed size of the `num_tracks` + `num_uids` pre-amble in bytes
    /// (§1.1). `ckSize == PREAMBLE_LEN + N * AudioId::SIZE`.
    pub const PREAMBLE_LEN: usize = 4;

    /// Decode a `chna` chunk body. Returns `None` when the body is
    /// shorter than the 4-byte `num_tracks`+`num_uids` pre-amble. The
    /// record count is derived from the body length
    /// (`N = (body.len() - 4) / 40`); any trailing bytes that do not
    /// fill a whole 40-byte record are ignored (§1.1, §1.3).
    pub fn parse(buf: &[u8]) -> Option<ChnaChunk> {
        if buf.len() < Self::PREAMBLE_LEN {
            return None;
        }
        let num_tracks = u16::from_le_bytes([buf[0], buf[1]]);
        let num_uids = u16::from_le_bytes([buf[2], buf[3]]);
        let n = (buf.len() - Self::PREAMBLE_LEN) / AudioId::SIZE;
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let off = Self::PREAMBLE_LEN + i * AudioId::SIZE;
            // `AudioId::parse` cannot fail here — the slice is exactly
            // `SIZE` bytes by construction of `n`.
            if let Some(rec) = AudioId::parse(&buf[off..off + AudioId::SIZE]) {
                ids.push(rec);
            }
        }
        Some(ChnaChunk {
            num_tracks,
            num_uids,
            ids,
        })
    }

    /// On-disk size of the chunk **data section** in bytes (the `ckSize`
    /// value, excluding the 8-byte `ckID`+`ckSize` header) — §1.1.
    pub fn body_len(&self) -> usize {
        Self::PREAMBLE_LEN + self.ids.len() * AudioId::SIZE
    }

    /// Serialize the `chna` chunk body (`num_tracks`, `num_uids`, then
    /// every `audioID` record). The body is always even-sized so no
    /// inter-chunk pad byte is needed (§1.4).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.body_len());
        out.extend_from_slice(&self.num_tracks.to_le_bytes());
        out.extend_from_slice(&self.num_uids.to_le_bytes());
        for rec in &self.ids {
            out.extend_from_slice(&rec.to_bytes());
        }
        out
    }

    /// Iterator over the *defined* (non-spare) records — those whose
    /// `track_index != 0` (§1.3).
    pub fn defined_ids(&self) -> impl Iterator<Item = &AudioId> {
        self.ids.iter().filter(|r| !r.is_unused())
    }
}

/// Parse a `chna` chunk body, surface `wav:chna.*` metadata keys and
/// return the typed view for [`WavDemuxer::chna`]. Keys:
///
/// - `wav:chna.num_tracks` / `wav:chna.num_uids` — the two pre-amble
///   counts.
/// - `wav:chna.record_count` — `N`, the total `audioID` records
///   (including spares); equals `(body_len - 4) / 40`.
/// - `wav:chna.defined_count` — number of records with a non-zero
///   `track_index` (the in-use entries).
/// - `wav:chna.<n>.track_index` / `.uid` / `.track_ref` / `.pack_ref`
///   for every *defined* record, zero-based `<n>` by encounter order
///   (spare records are not surfaced individually). `.uid` /
///   `.track_ref` / `.pack_ref` are emitted only when the field is not
///   entirely NUL.
/// - `wav:chna.body_len` — only when the on-wire body exceeds the
///   record-aligned size (trailing extension bytes riding along).
fn parse_chna_chunk(buf: &[u8], out: &mut Vec<(String, String)>) -> Option<ChnaChunk> {
    let chna = ChnaChunk::parse(buf)?;
    out.push((
        "wav:chna.num_tracks".to_string(),
        chna.num_tracks.to_string(),
    ));
    out.push(("wav:chna.num_uids".to_string(), chna.num_uids.to_string()));
    out.push((
        "wav:chna.record_count".to_string(),
        chna.ids.len().to_string(),
    ));
    let defined = chna.defined_ids().count();
    out.push(("wav:chna.defined_count".to_string(), defined.to_string()));
    for (n, rec) in chna.defined_ids().enumerate() {
        out.push((
            format!("wav:chna.{n}.track_index"),
            rec.track_index.to_string(),
        ));
        if let Some(uid) = audio_id_text(&rec.uid) {
            out.push((format!("wav:chna.{n}.uid"), uid));
        }
        if let Some(tref) = audio_id_text(&rec.track_ref) {
            out.push((format!("wav:chna.{n}.track_ref"), tref));
            // §1.2: the trackRef names an audioTrackFormatID (`AT_`) or,
            // for linear-PCM essence, an audioChannelFormat (`AC_`).
            let kind = rec.track_ref_kind();
            if kind != AdmRefKind::Unknown {
                out.push((format!("wav:chna.{n}.track_ref_kind"), kind.as_str().into()));
            }
            // §3: BS.2094 common definition vs file-local custom one.
            if let Some(scope) = rec.track_ref_scope() {
                out.push((
                    format!("wav:chna.{n}.track_ref_definition"),
                    scope.as_str().into(),
                ));
            }
        }
        if let Some(pref) = audio_id_text(&rec.pack_ref) {
            out.push((format!("wav:chna.{n}.pack_ref"), pref));
            let kind = rec.pack_ref_kind();
            if kind != AdmRefKind::Unknown {
                out.push((format!("wav:chna.{n}.pack_ref_kind"), kind.as_str().into()));
            }
            if let Some(scope) = rec.pack_ref_scope() {
                out.push((
                    format!("wav:chna.{n}.pack_ref_definition"),
                    scope.as_str().into(),
                ));
            }
        }
    }
    // The §1.1 record count `N` is derived from the floor division, so a
    // body whose length isn't `4 + N*40` carries trailing extension
    // bytes; surface the raw on-wire length so they're observable.
    if buf.len() > chna.body_len() {
        out.push(("wav:chna.body_len".to_string(), buf.len().to_string()));
    }
    Some(chna)
}

/// Parse a `fact` chunk body per
/// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "FACT Chunk":
///
/// ```text
/// <fact-ck> -> fact( <dwFileSize:DWORD> )   // Number of samples per channel
/// ```
///
/// The 1991 RIFF MCI spec defines exactly one field — `dwFileSize`,
/// the count of *samples per channel* (the spec text uses "sample"
/// in the per-channel sense; the field name predates the
/// per-channel/per-frame disambiguation). The spec explicitly
/// reserves the trailing bytes for future extension fields: any
/// bytes past offset 4 must be tolerated, with the chunk size
/// telling applications which fields are present.
///
/// The `fact` chunk is required for compressed WAV streams and for
/// any file using the `wavl LIST` waveform container; for plain
/// PCM `data` chunks it is optional. The parser surfaces:
///
/// - `wav:fact.sample_count` — `dwFileSize` as a decimal string.
/// - `wav:fact.body_len` — total chunk-body length, present when the
///   body exceeds the 4-byte fixed field so downstream tools can see
///   that future-extension bytes are riding along (the bytes themselves
///   are opaque to this parser by spec).
///
/// Returns the `dwFileSize` value so the demuxer can use it as the
/// authoritative `total_samples` for the stream's duration. A body
/// shorter than 4 bytes is treated as opaque-and-skipped (returns
/// `None`, no metadata key emitted).
fn parse_fact_chunk(buf: &[u8], out: &mut Vec<(String, String)>) -> Option<u32> {
    if buf.len() < 4 {
        return None;
    }
    let sample_count = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    out.push((
        "wav:fact.sample_count".to_string(),
        sample_count.to_string(),
    ));
    if buf.len() > 4 {
        out.push(("wav:fact.body_len".to_string(), buf.len().to_string()));
    }
    Some(sample_count)
}

/// Parse an `iXML` chunk body and surface its payload through the
/// metadata table.
///
/// The `iXML` chunk carries a UTF-8 XML document (third-party
/// production-recorder metadata block — `IXML_VERSION`, `PROJECT`,
/// `SCENE`, `TAKE`, `TAPE`, `NOTE`, `UBITS`, `FILE_UID`, a `BWF`
/// sub-group mirroring the `bext` fields and a `TRACK_LIST` of
/// per-track name / function / channel-index mappings). The schema
/// catalog is documented in
/// `docs/container/riff/metadata/exiftool-riff-tags.html` § `iXML`
/// and discussed in
/// `docs/container/riff/metadata/README.md` § "iXML".
///
/// The chunk body is surfaced verbatim under `wav:ixml` (trimmed at
/// the first NUL and stripped of surrounding whitespace so a writer
/// that pads the body to a fixed length with NULs does not surface
/// spurious trailing bytes). The raw chunk-body length is always
/// surfaced under `wav:ixml.body_len` whenever the chunk is present
/// — even when the text payload itself is empty — so downstream
/// tooling can distinguish "no `iXML` chunk" from "an `iXML` chunk
/// whose body is entirely NULs / whitespace".
///
/// An empty chunk body (zero bytes between header and pad) is treated
/// as opaque: `wav:ixml.body_len = 0` is emitted but no `wav:ixml`
/// text key is added.
fn parse_ixml_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    out.push(("wav:ixml.body_len".to_string(), buf.len().to_string()));
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let text = String::from_utf8_lossy(&buf[..end]).trim().to_string();
    if !text.is_empty() {
        out.push(("wav:ixml".to_string(), text));
    }
}

/// Parse an `<axml>` chunk body and surface its XML payload through
/// the metadata table.
///
/// The `<axml>` chunk carries an XML document compliant with XML 1.0
/// (or later), per
/// `docs/container/riff/metadata/ebu-tech3285s5-ADM.pdf` §3 "AXML
/// chunk definition":
///
/// > The <axml> chunk consists of a header followed by data compliant
/// > with the XML format. The overall length of the chunk is not fixed.
/// >
/// > typedef struct axml {
/// >     CHAR    ckID[4];     // {'a','x','m','l'}
/// >     DWORD   ckSize;      // size of chunk
/// >     CHAR    xmlData[];   // text data in XML
/// > } axml_chunk;
///
/// The §3 "Terminology" paragraph also notes the `<axml>` chunk may
/// occur in any order relative to other BWF chunks within the same
/// file — the demuxer's chunk-walk already tolerates any inter-chunk
/// ordering between `fmt ` and `data`.
///
/// Typical payloads are EBUCore (`<ebuCoreMain>`) wrappers around an
/// `<audioFormatExtended>` ADM document or an ISRC identifier
/// declaration (§4.1 + §4.2 examples). This parser does not interpret
/// the XML schema — it surfaces the textual payload verbatim so a
/// downstream tool (or a higher-level ADM-aware crate) can apply the
/// schema-specific decoding without re-walking the RIFF tree.
///
/// Surface shape mirrors `parse_ixml_chunk` (the sibling third-party
/// XML metadata block):
///
/// * `wav:axml.body_len` — raw on-wire chunk-body length. Always
///   emitted when the chunk is present (even for empty / NUL-only /
///   whitespace-only bodies) so downstream tooling can distinguish
///   "no `<axml>` chunk" from "an `<axml>` chunk reserved for later
///   population". Excludes the 8-byte chunk header and the implicit
///   RIFF §2 word-align pad byte.
/// * `wav:axml` — the UTF-8 XML text payload. Trimmed at the first
///   NUL byte (writers commonly NUL-pad to reserve room for in-place
///   editing of a larger ADM document) and stripped of surrounding
///   whitespace. Omitted entirely when the pre-NUL, trimmed text is
///   empty — the §3 note "if the receiving device cannot interpret
///   the content of the <axml> chunk in accordance with the
///   specification stated in the XML, the entire chunk shall be
///   ignored" applies to the *schema* level, not the byte level, so a
///   present-but-empty body still surfaces its `body_len` so the
///   placeholder is observable.
fn parse_axml_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    out.push(("wav:axml.body_len".to_string(), buf.len().to_string()));
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let text = String::from_utf8_lossy(&buf[..end]).trim().to_string();
    if !text.is_empty() {
        out.push(("wav:axml".to_string(), text));
    }
}

/// Parse a `<bxml>` chunk body and surface its (optionally compressed)
/// XML payload through the metadata table.
///
/// The `<bxml>` chunk is the compressed-XML counterpart of `<axml>`,
/// defined in ITU-R BS.2088-2 §6 "BXML chunk"
/// (`docs/container/riff/metadata/R-REC-BS.2088.pdf`):
///
/// > The <bxml> chunk may contain the compressed XML data instead of
/// > the <axml> chunk. The <bxml> chunk consists of a header followed
/// > by the XML data compressed by the compression method specified in
/// > the fmtType.
/// >
/// > struct bxml_chunk {
/// >     CHAR  ckID[4];     // {'b','x','m','l'}
/// >     DWORD ckSize;      // size of the <bxml> chunk in bytes
/// >     WORD  fmtType;     // type of compression method,
/// >                        // 0x0001="gzip", etc.
/// >     CHAR  xmlData[];   // XML text data compressed by the method
/// > };
///
/// Per §6.2 the `fmtType` value `0x0000` means the `xmlData` payload is
/// *uncompressed* XML text, while `0x0001` selects gzip (IETF RFC 1952).
/// The body may legitimately exceed 4 GiB (§6.1), in which case the
/// 32-bit on-wire `ckSize` carries the `0xFFFFFFFF` sentinel and the
/// true size rides in the `<ds64>` chunk's `table` array keyed on the
/// `bxml` chunk-ID — the demuxer's existing `ds64` sentinel-promotion
/// path already resolves this before the chunk body is read.
///
/// Surface shape mirrors `parse_axml_chunk` (its uncompressed sibling)
/// with the two compression-header fields added:
///
/// * `wav:bxml.body_len` — raw on-wire chunk-body length (includes the
///   2-byte `fmtType` header). Always emitted when the chunk is present
///   so a NUL-reserved placeholder block is observable, mirroring the
///   `<axml>` contract. Excludes the 8-byte chunk header and the RIFF
///   §2 word-align pad byte.
/// * `wav:bxml.fmt_type` — the raw 16-bit `fmtType` value as `0x%04X`.
///   Always emitted when the 2-byte header is present.
/// * `wav:bxml.compression` — a human-readable label for the documented
///   `fmtType` codes (`none` for `0x0000`, `gzip` for `0x0001`);
///   omitted for any other (private / future) compression code so the
///   raw `fmt_type` is the unambiguous source of truth.
/// * `wav:bxml` — the UTF-8 XML text payload, surfaced **only** when
///   `fmtType == 0x0000` (uncompressed). Trimmed at the first NUL byte
///   and stripped of surrounding whitespace, exactly like `<axml>`.
///   For compressed payloads the bytes are not decompressed here (no
///   schema interpretation at the container layer — a higher-level
///   ADM-aware consumer applies RFC 1952 inflation if needed), so only
///   `body_len` / `fmt_type` / `compression` are surfaced.
///
/// Bodies shorter than the 2-byte `fmtType` header are skipped as
/// opaque: only `wav:bxml.body_len` is emitted so the malformed/empty
/// placeholder remains observable.
fn parse_bxml_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    out.push(("wav:bxml.body_len".to_string(), buf.len().to_string()));
    if buf.len() < 2 {
        return;
    }
    let fmt_type = u16::from_le_bytes([buf[0], buf[1]]);
    out.push(("wav:bxml.fmt_type".to_string(), format!("0x{fmt_type:04X}")));
    match fmt_type {
        0x0000 => out.push(("wav:bxml.compression".to_string(), "none".to_string())),
        0x0001 => out.push(("wav:bxml.compression".to_string(), "gzip".to_string())),
        _ => {}
    }
    // The textual XML is only directly readable for the uncompressed
    // form; compressed payloads are surfaced via the header fields only.
    if fmt_type == 0x0000 {
        let payload = &buf[2..];
        let end = payload
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(payload.len());
        let text = String::from_utf8_lossy(&payload[..end]).trim().to_string();
        if !text.is_empty() {
            out.push(("wav:bxml".to_string(), text));
        }
    }
}

/// Parse a `_PMX` (Adobe XMP packet) chunk body and surface its UTF-8
/// XMP packet through the metadata table.
///
/// The `_PMX` FOURCC is the WAV/AVI carrier for an XMP serialised
/// packet, catalogued under
/// `docs/container/riff/metadata/exiftool-riff-tags.html` § "RIFF Main
/// tags" (entry `'_PMX'`, family `XMP`, scope note "AVI and WAV
/// files"). The FOURCC is little-endian "XMP_" reversed — the same
/// convention RIFF uses for chunks whose payload originates in a
/// little-endian DWORD-aligned authoring tool. The payload itself is
/// the XMP packet text exactly as it would appear in an XMP sidecar
/// (`x:xmpmeta` wrapped in `<?xpacket begin=...?>` /
/// `<?xpacket end=...?>` processing instructions).
///
/// Surface shape mirrors `parse_ixml_chunk` and `parse_axml_chunk`
/// (the two existing third-party XML metadata blocks) for orthogonality
/// at the consumer surface:
///
/// * `wav:xmp.body_len` — raw on-wire chunk-body length. Always
///   emitted when the `_PMX` chunk is present (even for empty /
///   NUL-only / whitespace-only bodies) so downstream tooling can
///   distinguish "no `_PMX` chunk" from "an `_PMX` chunk reserved for
///   later population by an XMP-aware writer". Excludes the 8-byte
///   chunk header and the implicit RIFF §2 word-align pad byte.
/// * `wav:xmp` — the UTF-8 XMP packet text. Trimmed at the first NUL
///   byte (writers commonly NUL-pad to a fixed length so an XMP-aware
///   editor can rewrite the packet in place without re-walking the
///   chunk graph) and stripped of surrounding whitespace. Omitted
///   entirely when the pre-NUL, trimmed text is empty — the
///   placeholder body length still surfaces so the reservation is
///   observable.
///
/// This parser deliberately does not interpret the XMP schema (RDF,
/// namespace prefixes, xpacket processing instructions). A higher
/// layer (or a dedicated XMP crate) can apply the schema-specific
/// decoding without re-walking the RIFF tree. Keeps the wav module's
/// surface schema-agnostic, matching the existing `_PMX`-adjacent
/// `iXML` and `<axml>` parsers.
fn parse_pmx_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    out.push(("wav:xmp.body_len".to_string(), buf.len().to_string()));
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let text = String::from_utf8_lossy(&buf[..end]).trim().to_string();
    if !text.is_empty() {
        out.push(("wav:xmp".to_string(), text));
    }
}

/// Surface metadata for a `JUNK` (Filler) chunk per
/// `docs/container/riff/metadata/microsoft-riffmci.pdf` §2 "JUNK
/// (Filler) Chunk":
///
/// > A JUNK chunk represents padding, filler or outdated information.
/// > It contains no relevant data; it is a space filler of arbitrary
/// > size.
///
/// We deliberately do not surface the chunk body (it's defined as
/// random/outdated data with no semantic content). What we do surface
/// is *accounting*: how many `JUNK` chunks the file contains and the
/// cumulative number of payload bytes they reserve. This lets a
/// downstream tool answer "is this writer leaving room for in-place
/// edits, and how much?" without us pretending the bytes carry meaning.
///
/// Keys:
///
/// * `wav:junk.count` — number of `JUNK` chunks seen so far. Each
///   call increments the counter so a multi-`JUNK` file (common when
///   a writer reserves separate slots ahead of `LIST INFO` and ahead
///   of `data`) is fully observable.
/// * `wav:junk.total_bytes` — cumulative payload size across all
///   `JUNK` chunks (the spec's "arbitrary size" filler region; does
///   not include the 8-byte chunk header or the trailing word-align
///   pad byte).
/// * `wav:junk.<n>.body_len` — per-chunk payload size, indexed
///   zero-based by encounter order. Allows a downstream tool to
///   distinguish "one big filler" from "many small fillers" without
///   re-walking the file.
///
/// Empty (`size = 0`) `JUNK` chunks are tolerated and still bump the
/// counter; their `body_len` surfaces as `0`. The spec calls the
/// filler "of arbitrary size" so a zero-length body is in-range.
fn surface_junk_metadata(out: &mut Vec<(String, String)>, size: u64) {
    // Count existing entries to derive the next zero-based index and
    // the running total. We linear-scan the metadata vector because
    // it's already keyed by string and we want a single source of
    // truth (no parallel counter to forget to update). Metadata
    // vectors stay small (low hundreds of entries) so this is O(n)
    // per JUNK chunk over a tiny n.
    let mut count: u64 = 0;
    let mut total: u64 = 0;
    for (k, v) in out.iter() {
        if k == "wav:junk.count" {
            count = v.parse().unwrap_or(count);
        } else if k == "wav:junk.total_bytes" {
            total = v.parse().unwrap_or(total);
        }
    }
    let idx = count;
    count = count.saturating_add(1);
    total = total.saturating_add(size);
    // Per-chunk entry.
    out.push((format!("wav:junk.{idx}.body_len"), size.to_string()));
    // Update the rolling aggregates. We push fresh entries rather than
    // mutate in place so the vector stays append-only (matching how
    // every other chunk parser in this module emits).
    out.retain(|(k, _)| k != "wav:junk.count" && k != "wav:junk.total_bytes");
    out.push(("wav:junk.count".to_string(), count.to_string()));
    out.push(("wav:junk.total_bytes".to_string(), total.to_string()));
}

/// Surface metadata for a `slnt` (silence) chunk per
/// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "Wave Data":
///
/// > `<silence-ck>` ➝ `slnt( <dwSamples:DWORD> )` — Count of silent
/// > samples.
///
/// The §3 note clarifies that "the `slnt` chunk represents silence, not
/// necessarily a repeated zero volume or baseline sample" — i.e. the
/// chunk records a *count of silent samples* rather than carrying any
/// PCM payload. A `slnt` chunk most commonly appears inside a `wavl`
/// LIST alternating with `data` chunks (a sparse-silence encoding), but
/// the §3 grammar `<wave-data> ➝ { <data-ck> | <data-list> }` also lets
/// the demuxer encounter a top-level `slnt` sibling of `data`; we
/// account for every occurrence either way.
///
/// We deliberately do not synthesise real zero/baseline samples into
/// the decoded stream (that is a host-runtime playback decision, and
/// the §3 note is explicit that the "right" fill value is
/// context-dependent — the last-played sample, not necessarily zero).
/// Instead we surface accounting so a downstream tool can observe how
/// many silent samples the producer encoded sparsely without re-walking
/// the file:
///
/// * `wav:slnt.count` — number of `slnt` chunks seen so far. Each call
///   increments the counter so a multi-`slnt` file (the normal `wavl`
///   case) is fully observable.
/// * `wav:slnt.total_samples` — cumulative silent-sample count across
///   all `slnt` chunks (the sum of every `dwSamples` field).
/// * `wav:slnt.<n>.samples` — per-chunk `dwSamples` value, indexed
///   zero-based by encounter order.
///
/// A body shorter than the 4-byte `dwSamples` field is treated as
/// opaque: the chunk is still counted (so the reservation is
/// observable) but contributes `0` to the running sample total and its
/// per-chunk `samples` key is omitted, mirroring how the other
/// fixed-struct parsers treat an under-length body.
fn surface_slnt_metadata(out: &mut Vec<(String, String)>, buf: &[u8]) {
    let mut count: u64 = 0;
    let mut total: u64 = 0;
    for (k, v) in out.iter() {
        if k == "wav:slnt.count" {
            count = v.parse().unwrap_or(count);
        } else if k == "wav:slnt.total_samples" {
            total = v.parse().unwrap_or(total);
        }
    }
    let idx = count;
    count = count.saturating_add(1);
    if buf.len() >= 4 {
        let samples = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        total = total.saturating_add(samples as u64);
        out.push((format!("wav:slnt.{idx}.samples"), samples.to_string()));
    }
    out.retain(|(k, _)| k != "wav:slnt.count" && k != "wav:slnt.total_samples");
    out.push(("wav:slnt.count".to_string(), count.to_string()));
    out.push(("wav:slnt.total_samples".to_string(), total.to_string()));
}

/// Map a Microsoft RIFF MCI §3 "Country Codes" three-digit code to its
/// human-readable name. Returns `None` for unknown codes; the caller
/// surfaces the numeric value regardless so a future code-page addition
/// is still visible.
fn cset_country_name(code: u16) -> Option<&'static str> {
    // Verbatim from `docs/container/riff/metadata/microsoft-riffmci.pdf`
    // §3 "Country Codes" (the table immediately following the CSET
    // definition). Codes are three-digit telephony region codes.
    match code {
        0 => Some("None"),
        1 => Some("USA"),
        2 => Some("Canada"),
        3 => Some("Latin America"),
        30 => Some("Greece"),
        31 => Some("Netherlands"),
        32 => Some("Belgium"),
        33 => Some("France"),
        34 => Some("Spain"),
        39 => Some("Italy"),
        41 => Some("Switzerland"),
        43 => Some("Austria"),
        44 => Some("United Kingdom"),
        45 => Some("Denmark"),
        46 => Some("Sweden"),
        47 => Some("Norway"),
        49 => Some("West Germany"),
        52 => Some("Mexico"),
        55 => Some("Brazil"),
        61 => Some("Australia"),
        64 => Some("New Zealand"),
        81 => Some("Japan"),
        82 => Some("Korea"),
        86 => Some("People's Republic of China"),
        88 => Some("Taiwan"),
        90 => Some("Turkey"),
        351 => Some("Portugal"),
        352 => Some("Luxembourg"),
        354 => Some("Iceland"),
        358 => Some("Finland"),
        _ => None,
    }
}

/// Map a Microsoft RIFF MCI §3 (`wLanguage`, `wDialect`) pair to its
/// human-readable name. Returns `None` for any unknown pair; the caller
/// surfaces the raw numeric values regardless so dialects added by
/// vendor extensions are still observable.
fn cset_language_name(language: u16, dialect: u16) -> Option<&'static str> {
    // Verbatim from `docs/container/riff/metadata/microsoft-riffmci.pdf`
    // §3 "Language and Dialect Codes". The dialect column disambiguates
    // regional variants (e.g. UK vs US English, Belgian vs Canadian
    // French, Latin vs Cyrillic Serbo-Croatian). A `(language, 0)` pair
    // is "ignore dialect": resolved as the first listed dialect when
    // present, otherwise `None`.
    match (language, dialect) {
        (0, _) => Some("None"),
        (1, 1) => Some("Arabic"),
        (2, 1) => Some("Bulgarian"),
        (3, 1) => Some("Catalan"),
        (4, 1) => Some("Traditional Chinese"),
        (4, 2) => Some("Simplified Chinese"),
        (5, 1) => Some("Czech"),
        (6, 1) => Some("Danish"),
        (7, 1) => Some("German"),
        (7, 2) => Some("Swiss German"),
        (8, 1) => Some("Greek"),
        (9, 1) => Some("US English"),
        (9, 2) => Some("UK English"),
        (10, 1) => Some("Spanish"),
        (10, 2) => Some("Spanish Mexican"),
        (11, 1) => Some("Finnish"),
        (12, 1) => Some("French"),
        (12, 2) => Some("Belgian French"),
        (12, 3) => Some("Canadian French"),
        (12, 4) => Some("Swiss French"),
        (13, 1) => Some("Hebrew"),
        (14, 1) => Some("Hungarian"),
        (15, 1) => Some("Icelandic"),
        (16, 1) => Some("Italian"),
        (16, 2) => Some("Swiss Italian"),
        (17, 1) => Some("Japanese"),
        (18, 1) => Some("Korean"),
        (19, 1) => Some("Dutch"),
        (19, 2) => Some("Belgian Dutch"),
        (20, 1) => Some("Norwegian - Bokmal"),
        (20, 2) => Some("Norwegian - Nynorsk"),
        (21, 1) => Some("Polish"),
        (22, 1) => Some("Brazilian Portuguese"),
        (22, 2) => Some("Portuguese"),
        (23, 1) => Some("Rhaeto-Romanic"),
        (24, 1) => Some("Romanian"),
        (25, 1) => Some("Russian"),
        (26, 1) => Some("Serbo-Croatian (Latin)"),
        (26, 2) => Some("Serbo-Croatian (Cyrillic)"),
        (27, 1) => Some("Slovak"),
        (28, 1) => Some("Albanian"),
        (29, 1) => Some("Swedish"),
        (30, 1) => Some("Thai"),
        (31, 1) => Some("Turkish"),
        (32, 1) => Some("Urdu"),
        (33, 1) => Some("Bahasa"),
        _ => None,
    }
}

/// Parse a `CSET` (Character Set) chunk body per
/// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3
/// "CSET (Character Set) Chunk":
///
/// ```text
/// <CSET-chunk> -> CSET( <wCodePage:WORD>
///                       <wCountryCode:WORD>
///                       <wLanguageCode:WORD>
///                       <wDialect:WORD> )
/// ```
///
/// All four fields are 16-bit little-endian; the chunk body is therefore
/// exactly 8 bytes in the canonical form. The CSET chunk declares the
/// code page, country, language and dialect that file elements (notably
/// the `LIST INFO` ZSTR sub-chunks) are interpreted under. Per the spec,
/// each field's zero value is "ignore" / "use the default":
///
/// * `wCodePage = 0` → ISO 8859/1 (Latin-1), identical to code page 1004
///   without hex columns 0/1/8/9.
/// * `wCountryCode = 0` → USA (country code 001).
/// * `wLanguageCode = 0` / `wDialect = 0` → US English (language 9,
///   dialect 1).
///
/// The parser surfaces:
///
/// * `wav:cset.code_page` — raw `wCodePage` decimal value (0 = ISO
///   8859/1; any non-zero value is the 16-bit Windows / OS-2 code-page
///   number, e.g. 1252 for Western European, 932 for Shift-JIS, 65001
///   for UTF-8).
/// * `wav:cset.country` — raw `wCountryCode` decimal value.
/// * `wav:cset.country_name` — human-readable name from the §3
///   "Country Codes" table (only when the code is in the spec's
///   enumerated set).
/// * `wav:cset.language` — raw `wLanguageCode` decimal value.
/// * `wav:cset.dialect` — raw `wDialect` decimal value.
/// * `wav:cset.language_name` — human-readable name from the §3
///   "Language and Dialect Codes" table (only when the pair is in the
///   spec's enumerated set).
/// * `wav:cset.body_len` — total chunk-body length, always emitted when
///   the chunk is present so a writer that grew the chunk for forward
///   compatibility is still observable.
///
/// Bodies shorter than the 8-byte canonical struct are treated as
/// opaque: only `wav:cset.body_len` is emitted. Bodies longer than 8
/// bytes have the trailing region tolerated (forward compatibility); the
/// `body_len` key lets downstream tooling notice the extra payload.
fn parse_cset_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    out.push(("wav:cset.body_len".to_string(), buf.len().to_string()));
    if buf.len() < 8 {
        return;
    }
    let code_page = u16::from_le_bytes([buf[0], buf[1]]);
    let country = u16::from_le_bytes([buf[2], buf[3]]);
    let language = u16::from_le_bytes([buf[4], buf[5]]);
    let dialect = u16::from_le_bytes([buf[6], buf[7]]);
    out.push(("wav:cset.code_page".to_string(), code_page.to_string()));
    out.push(("wav:cset.country".to_string(), country.to_string()));
    if let Some(name) = cset_country_name(country) {
        out.push(("wav:cset.country_name".to_string(), name.to_string()));
    }
    out.push(("wav:cset.language".to_string(), language.to_string()));
    out.push(("wav:cset.dialect".to_string(), dialect.to_string()));
    if let Some(name) = cset_language_name(language, dialect) {
        out.push(("wav:cset.language_name".to_string(), name.to_string()));
    }
}

/// Map a `LIST INFO` sub-chunk FOURCC to its conventional metadata-key
/// name, covering the full Microsoft RIFF MCI §3 "INFO List Chunk"
/// baseline (per `docs/container/riff/metadata/microsoft-riffmci.pdf`
/// pp. 2-14 .. 2-16). The 1991 baseline registers 23 sub-IDs; this
/// function returns the key under which each is surfaced to the
/// caller through `Demuxer::metadata`.
///
/// The mapping keeps the prior short-form names for the four widely-
/// quoted "audio tag" sub-IDs (`INAM` → `title`, `IART` → `artist`,
/// `IPRD` → `album`, `ICMT` → `comment`, `ICRD` → `date`, `IGNR` →
/// `genre`, `ICOP` → `copyright`, `IENG` → `engineer`, `ITCH` →
/// `technician`, `ISFT` → `encoder`, `ISBJ` → `subject`). `IPRD`'s
/// "album" alias is conventional rather than spec-literal — the §3
/// description says "Product. Specifies the name of the title the
/// file was originally intended for"; the tagging community has
/// long re-used the field to carry album names (matching how
/// ExifTool surfaces it).
///
/// The remaining baseline sub-IDs surface under spec-derived
/// snake_case names that come straight from the §3 entries:
///
/// * `IARL` Archival Location → `archival_location`.
/// * `ICMS` Commissioned → `commissioned`.
/// * `ICRP` Cropped → `cropped`.
/// * `IDIM` Dimensions → `dimensions`.
/// * `IDPI` Dots Per Inch → `dpi`.
/// * `IKEY` Keywords → `keywords`.
/// * `ILGT` Lightness → `lightness`.
/// * `IMED` Medium → `medium`.
/// * `IPLT` Palette Setting → `palette_setting`.
/// * `ISHP` Sharpness → `sharpness`.
/// * `ISRC` Source → `source`.
/// * `ISRF` Source Form → `source_form`.
///
/// `ITRK` (Track Number) is not in the §3 baseline but is the
/// canonical non-baseline tag every WAV-handling tool surfaces; we
/// keep it for compatibility with the existing public-API behaviour.
///
/// Beyond the 1991 baseline, this function also recognises the
/// **extended `INFO` sub-ID namespace** catalogued in
/// `docs/container/riff/metadata/exiftool-riff-tags.html`
/// (ExifTool's "RIFF Info Tags" table — the practical enumeration of
/// every `INFO`-LIST sub-ID encountered in production WAV/AVI files,
/// including the Microsoft "more info" / Windows-Media set, the
/// per-stream audio-language slots, and the common production-credit
/// tags). Each is a plain ZSTR text field exactly like the baseline
/// entries, so the parser surfaces it under a spec-derived snake_case
/// key taken from the ExifTool tag name. These remain distinct from
/// the baseline group so a caller can still tell a 1991-baseline tag
/// from a vendor extension by the documented key set.
///
/// Sub-IDs outside this set return `None`; their bytes are skipped
/// by `parse_info_list` rather than surfaced under a synthetic key.
fn info_id_to_key(id: &[u8; 4]) -> Option<&'static str> {
    match id {
        // RIFF MCI §3 baseline (1991), in spec order.
        b"IARL" => Some("archival_location"),
        b"IART" => Some("artist"),
        b"ICMS" => Some("commissioned"),
        b"ICMT" => Some("comment"),
        b"ICOP" => Some("copyright"),
        b"ICRD" => Some("date"),
        b"ICRP" => Some("cropped"),
        b"IDIM" => Some("dimensions"),
        b"IDPI" => Some("dpi"),
        b"IENG" => Some("engineer"),
        b"IGNR" => Some("genre"),
        b"IKEY" => Some("keywords"),
        b"ILGT" => Some("lightness"),
        b"IMED" => Some("medium"),
        b"INAM" => Some("title"),
        b"IPLT" => Some("palette_setting"),
        b"IPRD" => Some("album"),
        b"ISBJ" => Some("subject"),
        b"ISFT" => Some("encoder"),
        b"ISHP" => Some("sharpness"),
        b"ISRC" => Some("source"),
        b"ISRF" => Some("source_form"),
        b"ITCH" => Some("technician"),
        // Non-baseline but ubiquitous in tag-writer output.
        b"ITRK" => Some("track"),
        // Extended `INFO` sub-IDs catalogued in ExifTool's RIFF Info
        // Tags table (`exiftool-riff-tags.html`). Keys are the
        // snake_case form of the documented ExifTool tag name.
        b"IAS1" => Some("first_language"),
        b"IAS2" => Some("second_language"),
        b"IAS3" => Some("third_language"),
        b"IAS4" => Some("fourth_language"),
        b"IAS5" => Some("fifth_language"),
        b"IAS6" => Some("sixth_language"),
        b"IAS7" => Some("seventh_language"),
        b"IAS8" => Some("eighth_language"),
        b"IAS9" => Some("ninth_language"),
        b"IBSU" => Some("base_url"),
        b"ICAS" => Some("default_audio_stream"),
        b"ICDS" => Some("costume_designer"),
        b"ICNM" => Some("cinematographer"),
        b"ICNT" => Some("country"),
        b"IDIT" => Some("date_time_original"),
        b"IDST" => Some("distributed_by"),
        b"IEDT" => Some("edited_by"),
        b"IENC" => Some("encoded_by"),
        b"ILGU" => Some("logo_url"),
        b"ILIU" => Some("logo_icon_url"),
        b"ILNG" => Some("language"),
        b"IMBI" => Some("more_info_banner_image"),
        b"IMBU" => Some("more_info_banner_url"),
        b"IMIT" => Some("more_info_text"),
        b"IMIU" => Some("more_info_url"),
        b"IMUS" => Some("music_by"),
        b"IPDS" => Some("production_designer"),
        b"IPRO" => Some("produced_by"),
        b"IRIP" => Some("ripped_by"),
        b"IRTD" => Some("rating"),
        b"ISGN" => Some("secondary_genre"),
        b"ISMP" => Some("time_code"),
        b"ISTD" => Some("production_studio"),
        b"ISTR" => Some("starring"),
        b"IWMU" => Some("watermark_url"),
        b"IWRI" => Some("written_by"),
        _ => None,
    }
}

/// Fixed (pre-`CodingHistory`) size of the BWF `bext` struct, in bytes.
///
/// Sum of the field widths from EBU Tech 3285 v2 §2.3 `BROADCAST_EXT`:
/// `Description[256] + Originator[32] + OriginatorReference[32] +
/// OriginationDate[10] + OriginationTime[8] + TimeReferenceLow(4) +
/// TimeReferenceHigh(4) + Version(2) + UMID[64] + LoudnessValue(2) +
/// LoudnessRange(2) + MaxTruePeakLevel(2) + MaxMomentaryLoudness(2) +
/// MaxShortTermLoudness(2) + Reserved[180]` = 602.
const BEXT_FIXED_LEN: usize = 602;

/// Typed view of a BWF `bext` (Broadcast Audio Extension) chunk
/// (EBU Tech 3285 v2 §2.3 `BROADCAST_EXT`).
///
/// Layout per `docs/container/riff/metadata/ebu-tech3285-bwf.pdf`. The
/// fixed struct is always [`BEXT_FIXED_LEN`] (602) bytes regardless of
/// `version`; `coding_history` is the variable-length ASCII tail past
/// the fixed struct (`= chunk size − 602`). All multi-byte integers are
/// little-endian (RIFF convention).
///
/// `version` selects which fields are meaningful: v0 populates none of
/// the UMID / loudness fields, v1 adds the SMPTE-330M UMID, v2 adds the
/// five loudness WORDs (§1.1). The struct stores every field verbatim
/// regardless of version so a read→write pass is byte-lossless; the
/// loudness WORDs are kept as the raw signed `round(100 × value)`
/// integers (§2.4), not the rendered two-decimal form.
///
/// The ASCII string fields ([`Self::description`] etc.) are stored as
/// owned `String`s trimmed at the first NUL. On serialization
/// ([`Self::to_bytes`]) each is re-emitted into its fixed-width slot,
/// truncated to fit and NUL-padded to the field width per §2.3.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BextChunk {
    /// `Description[256]` — free ASCII description of the sequence.
    pub description: String,
    /// `Originator[32]` — name of the originator / producer.
    pub originator: String,
    /// `OriginatorReference[32]` — unambiguous reference per EBU R99.
    pub originator_reference: String,
    /// `OriginationDate[10]` — `YYYY-MM-DD` (or `YYYY_MM_DD`).
    pub origination_date: String,
    /// `OriginationTime[8]` — `HH:MM:SS` (or `HH_MM_SS`).
    pub origination_time: String,
    /// `TimeReference` — 64-bit sample count since midnight (reassembled
    /// from the low/high 32-bit halves).
    pub time_reference: u64,
    /// `Version` of the BWF (0, 1 or 2).
    pub version: u16,
    /// `UMID[64]` — SMPTE-330M Unique Material Identifier. All-zero when
    /// absent (v0). A 32-byte "basic UMID" zero-pads the trailing half.
    pub umid: [u8; 64],
    /// `LoudnessValue` (×100, signed) — meaningful only for v2.
    pub loudness_value: i16,
    /// `LoudnessRange` (×100, signed) — meaningful only for v2.
    pub loudness_range: i16,
    /// `MaxTruePeakLevel` (dBTP ×100, signed) — meaningful only for v2.
    pub max_true_peak_level: i16,
    /// `MaxMomentaryLoudness` (×100, signed) — meaningful only for v2.
    pub max_momentary_loudness: i16,
    /// `MaxShortTermLoudness` (×100, signed) — meaningful only for v2.
    pub max_short_term_loudness: i16,
    /// `CodingHistory` — variable-length ASCII tail (CR/LF-separated
    /// `A=…,F=…,…` lines per §2.3 / EBU R98). Empty when absent.
    pub coding_history: String,
}

impl Default for BextChunk {
    /// An empty v0 `bext`: all strings empty, `version` 0, zero UMID /
    /// loudness / time-reference. Serializes to the 602-byte fixed
    /// struct with no `CodingHistory` tail.
    fn default() -> Self {
        BextChunk {
            description: String::new(),
            originator: String::new(),
            originator_reference: String::new(),
            origination_date: String::new(),
            origination_time: String::new(),
            time_reference: 0,
            version: 0,
            umid: [0u8; 64],
            loudness_value: 0,
            loudness_range: 0,
            max_true_peak_level: 0,
            max_momentary_loudness: 0,
            max_short_term_loudness: 0,
            coding_history: String::new(),
        }
    }
}

impl BextChunk {
    /// Fixed body length of the `bext` chunk (pre-`CodingHistory`).
    pub const FIXED_LEN: usize = BEXT_FIXED_LEN;

    /// Decode a `bext` chunk body. Returns `None` when the body is
    /// shorter than the 602-byte fixed struct (treated as opaque, same
    /// policy as the other fixed-layout metadata chunks). Bytes past the
    /// fixed struct are the `CodingHistory` tail.
    pub fn parse(buf: &[u8]) -> Option<BextChunk> {
        if buf.len() < BEXT_FIXED_LEN {
            return None;
        }
        let time_ref_low = u32::from_le_bytes([buf[338], buf[339], buf[340], buf[341]]);
        let time_ref_high = u32::from_le_bytes([buf[342], buf[343], buf[344], buf[345]]);
        let mut umid = [0u8; 64];
        umid.copy_from_slice(&buf[348..412]);
        let r16 = |o: usize| -> i16 { i16::from_le_bytes([buf[o], buf[o + 1]]) };
        let coding_history = if buf.len() > BEXT_FIXED_LEN {
            bext_field(&buf[BEXT_FIXED_LEN..])
        } else {
            String::new()
        };
        Some(BextChunk {
            description: bext_field(&buf[0..256]),
            originator: bext_field(&buf[256..288]),
            originator_reference: bext_field(&buf[288..320]),
            origination_date: bext_field(&buf[320..330]),
            origination_time: bext_field(&buf[330..338]),
            time_reference: ((time_ref_high as u64) << 32) | (time_ref_low as u64),
            version: u16::from_le_bytes([buf[346], buf[347]]),
            umid,
            loudness_value: r16(412),
            loudness_range: r16(414),
            max_true_peak_level: r16(416),
            max_momentary_loudness: r16(418),
            max_short_term_loudness: r16(420),
            coding_history,
        })
    }

    /// Serialize the `bext` chunk body: the 602-byte fixed struct
    /// followed by the `CodingHistory` tail (no trailing NUL is added —
    /// EBU R98 lines already end CR/LF and the chunk size delimits the
    /// tail). String fields are truncated to their slot width and
    /// NUL-padded per §2.3.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; BEXT_FIXED_LEN];
        fn put_ascii(out: &mut [u8], offset: usize, width: usize, s: &str) {
            let bytes = s.as_bytes();
            let n = bytes.len().min(width);
            out[offset..offset + n].copy_from_slice(&bytes[..n]);
            // remaining bytes stay zero (NUL pad)
        }
        put_ascii(&mut out, 0, 256, &self.description);
        put_ascii(&mut out, 256, 32, &self.originator);
        put_ascii(&mut out, 288, 32, &self.originator_reference);
        put_ascii(&mut out, 320, 10, &self.origination_date);
        put_ascii(&mut out, 330, 8, &self.origination_time);
        out[338..342].copy_from_slice(&(self.time_reference as u32).to_le_bytes());
        out[342..346].copy_from_slice(&((self.time_reference >> 32) as u32).to_le_bytes());
        out[346..348].copy_from_slice(&self.version.to_le_bytes());
        out[348..412].copy_from_slice(&self.umid);
        out[412..414].copy_from_slice(&self.loudness_value.to_le_bytes());
        out[414..416].copy_from_slice(&self.loudness_range.to_le_bytes());
        out[416..418].copy_from_slice(&self.max_true_peak_level.to_le_bytes());
        out[418..420].copy_from_slice(&self.max_momentary_loudness.to_le_bytes());
        out[420..422].copy_from_slice(&self.max_short_term_loudness.to_le_bytes());
        // out[422..602] is Reserved[180] — left zero per §2.3.
        if !self.coding_history.is_empty() {
            out.extend_from_slice(self.coding_history.as_bytes());
        }
        out
    }
}

/// Trim a fixed-width ASCII field to its value: cut at the first NUL
/// (EBU Tech 3285 v2 §2.3 mandates a NUL terminator for under-length
/// strings) and strip surrounding whitespace.
fn bext_field(raw: &[u8]) -> String {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).trim().to_string()
}

/// Parse a BWF `bext` (Broadcast Audio Extension) chunk body and append
/// its fields to `out` under `wav:bext.*` keys.
///
/// Layout per `docs/container/riff/metadata/ebu-tech3285-bwf.pdf`
/// (EBU Tech 3285 v2 §2.3, `BROADCAST_EXT` struct). All multi-byte
/// integers are little-endian (RIFF convention). The loudness fields
/// (`LoudnessValue` … `MaxShortTermLoudness`) are signed 16-bit values
/// equal to `round(100 × value)` per §2.4, so they are surfaced divided
/// by 100 with two decimal places.
///
/// The `Version` field selects which fields are meaningful: v0 has none
/// of the UMID/loudness fields populated, v1 adds the SMPTE-330M UMID,
/// v2 adds the five loudness values (§1.1). The fixed struct is always
/// 602 bytes regardless of version, so this parser reads every field
/// unconditionally and lets the version key tell the caller which ones
/// to trust. `TimeReference` is reassembled as a 64-bit sample count.
fn parse_bext_chunk(buf: &[u8], out: &mut Vec<(String, String)>) -> Option<BextChunk> {
    let bext = BextChunk::parse(buf)?;

    let push = |out: &mut Vec<(String, String)>, key: &str, value: &str| {
        if !value.is_empty() {
            out.push((key.to_string(), value.to_string()));
        }
    };

    push(out, "wav:bext.description", &bext.description);
    push(out, "wav:bext.originator", &bext.originator);
    push(
        out,
        "wav:bext.originator_reference",
        &bext.originator_reference,
    );
    push(out, "wav:bext.origination_date", &bext.origination_date);
    push(out, "wav:bext.origination_time", &bext.origination_time);
    out.push((
        "wav:bext.time_reference".to_string(),
        bext.time_reference.to_string(),
    ));
    out.push(("wav:bext.version".to_string(), bext.version.to_string()));

    // v1+ : the SMPTE-330M UMID (64 bytes; a 32-byte "basic UMID"
    // zero-pads the trailing half per §2.3). Emit only when present.
    if bext.umid.iter().any(|&b| b != 0) {
        let mut hex = String::with_capacity(bext.umid.len() * 2);
        for b in &bext.umid {
            hex.push_str(&format!("{b:02x}"));
        }
        out.push(("wav:bext.umid".to_string(), hex));
    }

    // v2 : loudness metadata (×100 fixed-point → two decimals). Emitted
    // only for v2 files since v0/v1 leave these WORDs zero by spec.
    if bext.version >= 2 {
        out.push((
            "wav:bext.loudness_value".to_string(),
            fmt_loudness(bext.loudness_value),
        ));
        out.push((
            "wav:bext.loudness_range".to_string(),
            fmt_loudness(bext.loudness_range),
        ));
        out.push((
            "wav:bext.max_true_peak_level".to_string(),
            fmt_loudness(bext.max_true_peak_level),
        ));
        out.push((
            "wav:bext.max_momentary_loudness".to_string(),
            fmt_loudness(bext.max_momentary_loudness),
        ));
        out.push((
            "wav:bext.max_short_term_loudness".to_string(),
            fmt_loudness(bext.max_short_term_loudness),
        ));
    }

    push(out, "wav:bext.coding_history", &bext.coding_history);
    Some(bext)
}

/// Render a BWF loudness WORD (signed 16-bit, `round(100 × value)` per
/// EBU Tech 3285 v2 §2.4) back to its two-decimal value, e.g. `-2264`
/// → `"-22.64"`. The sign is carried on the integer part so values in
/// `(-1, 0)` keep their leading minus (`-50` → `"-0.50"`).
fn fmt_loudness(v: i16) -> String {
    let neg = v < 0;
    let abs = (v as i32).unsigned_abs();
    let whole = abs / 100;
    let frac = abs % 100;
    if neg {
        format!("-{whole}.{frac:02}")
    } else {
        format!("{whole}.{frac:02}")
    }
}

#[derive(Clone, Debug)]
struct WaveFmt {
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    #[allow(dead_code)]
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    /// `wValidBitsPerSample` from the 22-byte EXTENSIBLE extension.
    /// `None` for plain `WAVEFORMATEX`; `Some(0)` is a writer that left
    /// the union zero — the demuxer falls back to `bits_per_sample`.
    valid_bits_per_sample: Option<u16>,
    /// `dwChannelMask` — SPEAKER_* bitmap. `None` outside EXTENSIBLE.
    channel_mask: Option<u32>,
    /// 16-byte SubFormat GUID. `None` outside EXTENSIBLE.
    subformat: Option<[u8; 16]>,
}

fn parse_fmt(buf: &[u8]) -> Result<WaveFmt> {
    if buf.len() < 16 {
        return Err(Error::invalid("fmt chunk too small"));
    }
    let format_tag = u16::from_le_bytes([buf[0], buf[1]]);
    let channels = u16::from_le_bytes([buf[2], buf[3]]);
    let sample_rate = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let byte_rate = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let block_align = u16::from_le_bytes([buf[12], buf[13]]);
    let bits_per_sample = u16::from_le_bytes([buf[14], buf[15]]);
    let mut valid_bits_per_sample = None;
    let mut channel_mask = None;
    let mut subformat = None;
    if format_tag == FMT_EXTENSIBLE {
        // WAVEFORMATEXTENSIBLE layout per
        // docs/container/riff/waveformatextensible/README.md
        // §"Structure layout":
        //   [16..18]  cbSize             (must be >= 22)
        //   [18..20]  wValidBitsPerSample (active union member)
        //   [20..24]  dwChannelMask
        //   [24..40]  SubFormat GUID
        if buf.len() < 40 {
            return Err(Error::invalid(
                "WAVE_FORMAT_EXTENSIBLE fmt chunk shorter than the 40 bytes \
                 mandated by mmreg.h",
            ));
        }
        let cb_size = u16::from_le_bytes([buf[16], buf[17]]);
        if (cb_size as usize) < 22 {
            return Err(Error::invalid(format!(
                "WAVE_FORMAT_EXTENSIBLE cbSize must be >= 22, got {cb_size}"
            )));
        }
        valid_bits_per_sample = Some(u16::from_le_bytes([buf[18], buf[19]]));
        channel_mask = Some(u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]));
        let mut g = [0u8; 16];
        g.copy_from_slice(&buf[24..40]);
        subformat = Some(g);
    }
    Ok(WaveFmt {
        format_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        valid_bits_per_sample,
        channel_mask,
        subformat,
    })
}

/// Resolve a legacy `wFormatTag` + decode precision to a concrete PCM /
/// G.711 codec id, or `None` if the tag is not one of the formats this
/// crate maps directly. `bits` is the active precision (the container
/// `wBitsPerSample` on the legacy path, or the EXTENSIBLE union's
/// `wValidBitsPerSample` when that is non-zero).
///
/// Shared by the legacy `WAVEFORMATEX` path and the EXTENSIBLE path: per
/// `docs/container/riff/waveformatextensible/ms-converting-format-tags-and-subformat-guids.md`,
/// a SubFormat GUID built from `DEFINE_WAVEFORMATEX_GUID(x)` is exactly
/// equivalent to the legacy tag `x`, so both routes dispatch identically.
fn codec_for_tag(tag: u16, bits: u16) -> Result<Option<CodecId>> {
    Ok(match tag {
        FMT_PCM => Some(CodecId::new(pcm_int_codec(bits)?)),
        FMT_IEEE_FLOAT => Some(CodecId::new(pcm_float_codec(bits)?)),
        FMT_ALAW => Some(CodecId::new("pcm_alaw")),
        FMT_MULAW => Some(CodecId::new("pcm_mulaw")),
        _ => None,
    })
}

fn resolve_codec(fmt: &WaveFmt) -> Result<CodecId> {
    if fmt.format_tag != FMT_EXTENSIBLE {
        return match codec_for_tag(fmt.format_tag, fmt.bits_per_sample)? {
            Some(id) => Ok(id),
            None => Err(Error::unsupported(format!(
                "unsupported WAV format tag 0x{:04x}",
                fmt.format_tag
            ))),
        };
    }

    let sub = fmt
        .subformat
        .ok_or_else(|| Error::invalid("extensible WAV missing subformat"))?;
    // Per docs/container/riff/waveformatextensible/README.md
    // §"On using these specs": the actual codec precision is the
    // SubFormat union's wValidBitsPerSample, NOT the WAVEFORMATEX
    // wBitsPerSample (the container size). Fall back to the
    // container size only when the union is zero — some writers
    // leave it unset.
    let depth = fmt
        .valid_bits_per_sample
        .filter(|&v| v > 0)
        .unwrap_or(fmt.bits_per_sample);
    // Any SubFormat GUID built from the `KSMedia.h`
    // `DEFINE_WAVEFORMATEX_GUID(x)` template carries the legacy
    // `wFormatTag` `x` in its leading 16 bits and is, per
    // ms-converting-format-tags-and-subformat-guids.md, equivalent to the
    // legacy tag — so it dispatches through the SAME `codec_for_tag`
    // path the `WAVEFORMATEX` route uses. This generalises the four
    // hand-listed GUID constants (PCM 0x0001 / IEEE_FLOAT 0x0003 /
    // ALAW 0x0006 / MULAW 0x0007) to every tag-derived GUID. The
    // recursion guard (`tag != FMT_EXTENSIBLE`) rejects the degenerate
    // GUID whose embedded tag is 0xFFFE itself.
    if let Some(tag) = waveformatex_tag(&sub) {
        if tag != FMT_EXTENSIBLE {
            if let Some(id) = codec_for_tag(tag, depth)? {
                return Ok(id);
            }
        }
    }
    // Not a (mappable) WAVEFORMATEX-template GUID — synthesise a
    // `wav:guid_<text>` id so downstream make_decoder fails naming the
    // actual GUID rather than the opaque 0xFFFE tag. Mirrors the
    // `avi:guid_<...>` pattern in oxideav-avi.
    Ok(CodecId::new(format!("wav:guid_{}", fmt_guid(&sub))))
}

fn pcm_int_codec(bits: u16) -> Result<&'static str> {
    Ok(match bits {
        8 => "pcm_u8",
        16 => "pcm_s16le",
        24 => "pcm_s24le",
        32 => "pcm_s32le",
        _ => {
            return Err(Error::unsupported(format!(
                "unsupported WAV integer-PCM bit depth: {bits}"
            )));
        }
    })
}

fn pcm_float_codec(bits: u16) -> Result<&'static str> {
    Ok(match bits {
        32 => "pcm_f32le",
        64 => "pcm_f64le",
        _ => {
            return Err(Error::unsupported(format!(
                "unsupported WAV IEEE-float bit depth: {bits}"
            )));
        }
    })
}

/// Decoded sample-format hint for a WAV codec id. The host runtime
/// applies the actual decode (A-law/μ-law through `oxideav-g711`); this
/// crate only resolves what the *output* of that decode looks like in
/// the standard `SampleFormat` enum so callers building pipelines know
/// the shape ahead of time.
///
/// For unknown `wav:guid_<...>` ids the function returns `None` — the
/// EXTENSIBLE GUID didn't match any of the well-known SubFormats and
/// we don't pretend to know how the codec is decoded.
fn decoded_sample_format(id: &CodecId) -> Option<SampleFormat> {
    // Plain PCM codec ids forward to the existing helper.
    if let Some(fmt) = super::pcm::sample_format_for(id) {
        return Some(fmt);
    }
    match id.as_str() {
        // G.711 expands to S16 once decoded (oxideav-g711 alaw/mulaw
        // output is S16, matching the round-75 audio_codec_sample_format
        // mapping in oxideav-avi).
        "pcm_alaw" | "pcm_mulaw" => Some(SampleFormat::S16),
        _ => None,
    }
}

/// WAV demuxer.
///
/// Beyond the `Demuxer` trait, this type exposes round-77 accessors
/// for the `WAVEFORMATEXTENSIBLE` side-info — `format_tag`,
/// `valid_bits_per_sample`, `channel_mask`, `subformat`. Callers that
/// only have a `Box<dyn Demuxer>` should rely on the `wav:fmt.*`
/// metadata keys instead.
pub struct WavDemuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    data_offset: u64,
    data_end: u64,
    cursor: u64,
    block_align: u64,
    chunk_frames: u64,
    samples_emitted: i64,
    metadata: Vec<(String, String)>,
    duration_micros: i64,
    format_tag: u16,
    valid_bits_per_sample: Option<u16>,
    channel_mask: Option<u32>,
    subformat: Option<[u8; 16]>,
    acid: Option<AcidChunk>,
    chna: Option<ChnaChunk>,
    bext: Option<BextChunk>,
    cue: Option<CueChunk>,
    plst: Option<PlaylistChunk>,
    adtl: Option<AdtlChunk>,
}

impl WavDemuxer {
    /// On-wire `wFormatTag` from the `fmt ` chunk (one of `WAVE_FORMAT_*`).
    /// Preserved verbatim for round-trip purposes — the codec id
    /// already encodes the decoder dispatch.
    pub fn format_tag(&self) -> u16 {
        self.format_tag
    }

    /// `WAVEFORMATEXTENSIBLE.Samples.wValidBitsPerSample` — actual bit
    /// precision per sample. `None` for legacy `WAVEFORMATEX` streams
    /// (non-EXTENSIBLE `wFormatTag`).
    pub fn valid_bits_per_sample(&self) -> Option<u16> {
        self.valid_bits_per_sample
    }

    /// `WAVEFORMATEXTENSIBLE.dwChannelMask` — `SPEAKER_*` bitmap
    /// describing the channel ordering of the interleaved PCM byte
    /// stream. `None` for non-EXTENSIBLE streams.
    ///
    /// See `docs/container/riff/waveformatextensible/README.md`
    /// §"dwChannelMask bits" for the standard layouts.
    pub fn channel_mask(&self) -> Option<u32> {
        self.channel_mask
    }

    /// Human-readable speaker layout decoded from
    /// [`Self::channel_mask`] — a `+`-separated list of `SPEAKER_*`
    /// positions in canonical least-significant-bit-first order
    /// (e.g. `"FRONT_LEFT+FRONT_RIGHT+FRONT_CENTER+LOW_FREQUENCY"`).
    ///
    /// `None` when the stream is non-EXTENSIBLE, or when the mask is `0`
    /// (no assigned speaker positions). The same value is mirrored under
    /// the `wav:fmt.channel_layout` metadata key for `dyn Demuxer`
    /// consumers. See
    /// `docs/container/riff/waveformatextensible/ms-waveformatextensible.html`.
    pub fn channel_layout(&self) -> Option<String> {
        self.channel_mask.and_then(channel_mask_layout)
    }

    /// `WAVEFORMATEXTENSIBLE.SubFormat` — 16-byte GUID (the actual
    /// codec identifier when `format_tag == WAVE_FORMAT_EXTENSIBLE`).
    /// Returned in on-wire byte order (first three groups
    /// little-endian, trailing two groups big-endian); use
    /// [`Self::subformat_text`] for the canonical text representation.
    pub fn subformat(&self) -> Option<&[u8; 16]> {
        self.subformat.as_ref()
    }

    /// `WAVEFORMATEXTENSIBLE.SubFormat` formatted as a canonical
    /// `XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX` GUID string.
    pub fn subformat_text(&self) -> Option<String> {
        self.subformat.as_ref().map(fmt_guid)
    }

    /// Typed view of the Acidizer `acid` chunk when the file carried
    /// one with a well-formed 24-byte body. `None` when the chunk is
    /// absent or truncated. The same fields are mirrored under the
    /// `wav:acid.*` metadata keys for `dyn Demuxer` consumers.
    pub fn acid(&self) -> Option<&AcidChunk> {
        self.acid.as_ref()
    }

    /// Typed view of the BW64/ADM `chna` (channel-allocation) chunk
    /// when the file carried one with a well-formed body (ITU-R
    /// BS.2088-2 §8.1). `None` when the chunk is absent or shorter than
    /// the 4-byte count pre-amble. The same data is mirrored under the
    /// `wav:chna.*` metadata keys for `dyn Demuxer` consumers.
    pub fn chna(&self) -> Option<&ChnaChunk> {
        self.chna.as_ref()
    }

    /// Typed view of the BWF `bext` (Broadcast Audio Extension) chunk
    /// when the file carried one with a well-formed body (EBU Tech 3285
    /// v2 §2.3, ≥ 602-byte fixed struct). `None` when the chunk is
    /// absent or truncated. The same fields are mirrored under the
    /// `wav:bext.*` metadata keys for `dyn Demuxer` consumers; the typed
    /// view additionally exposes the raw loudness WORDs and the
    /// fixed-width UMID for round-trip muxing via
    /// [`WavMuxOptions::with_bext`].
    pub fn bext(&self) -> Option<&BextChunk> {
        self.bext.as_ref()
    }

    /// Typed view of the `cue ` (cue-points) chunk when the file
    /// carried one with a well-formed body (RIFF MCI §3 "Cue-Points
    /// Chunk"). `None` when the chunk is absent or shorter than the
    /// 4-byte count pre-amble. Chunks placed *after* the `data`
    /// waveform are read too — the same data is mirrored under the
    /// `wav:cue.*` metadata keys for `dyn Demuxer` consumers, and is
    /// re-emittable via [`WavMuxOptions::with_cue`].
    pub fn cue(&self) -> Option<&CueChunk> {
        self.cue.as_ref()
    }

    /// Typed view of the `plst` (playlist) chunk when the file carried
    /// one with a well-formed body (RIFF MCI §3 "Playlist Chunk").
    /// `None` when absent or truncated. Mirrored under the
    /// `wav:plst.*` metadata keys; re-emittable via
    /// [`WavMuxOptions::with_plst`].
    pub fn plst(&self) -> Option<&PlaylistChunk> {
        self.plst.as_ref()
    }

    /// Typed view of the `LIST adtl` (Associated Data List) chunk when
    /// the file carried one with at least one well-formed
    /// `labl`/`note`/`ltxt` entry (RIFF MCI §3 "Associated Data
    /// Chunk"). `None` when absent. Mirrored under the `wav:adtl.*`
    /// metadata keys; re-emittable via [`WavMuxOptions::with_adtl`].
    pub fn adtl(&self) -> Option<&AdtlChunk> {
        self.adtl.as_ref()
    }
}

impl Demuxer for WavDemuxer {
    fn format_name(&self) -> &str {
        "wav"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.cursor >= self.data_end {
            return Err(Error::Eof);
        }
        let remaining = self.data_end - self.cursor;
        let want_bytes = (self.chunk_frames * self.block_align).min(remaining);
        let want_bytes = (want_bytes / self.block_align) * self.block_align;
        if want_bytes == 0 {
            return Err(Error::Eof);
        }

        // Ensure we're positioned correctly (if an upstream operation seeked us).
        self.input.seek(SeekFrom::Start(self.cursor))?;
        let mut buf = vec![0u8; want_bytes as usize];
        self.input.read_exact(&mut buf)?;
        self.cursor += want_bytes;

        let stream = &self.streams[0];
        let frames = want_bytes / self.block_align;
        let pts = self.samples_emitted;
        self.samples_emitted += frames as i64;

        let mut pkt = Packet::new(0, stream.time_base, buf);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(frames as i64);
        pkt.flags.keyframe = true;
        Ok(pkt)
    }

    fn seek_to(&mut self, stream_index: u32, pts: i64) -> Result<i64> {
        if stream_index != 0 {
            return Err(Error::invalid(format!(
                "WAV: stream index {stream_index} out of range"
            )));
        }
        // PCM is keyframe-only and frame-aligned: the target pts is a
        // sample-index offset into the data chunk. Clamp to the valid
        // range and translate to a byte offset.
        let total_samples = (self.data_end - self.data_offset) / self.block_align;
        let target = (pts.max(0) as u64).min(total_samples);
        let new_cursor = self.data_offset + target * self.block_align;

        self.input.seek(SeekFrom::Start(new_cursor))?;
        self.cursor = new_cursor;
        self.samples_emitted = target as i64;
        Ok(target as i64)
    }

    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    fn duration_micros(&self) -> Option<i64> {
        if self.duration_micros > 0 {
            Some(self.duration_micros)
        } else {
            None
        }
    }
}

// --- Muxer -----------------------------------------------------------------

fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::unsupported("WAV supports exactly one audio stream"));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Audio {
        return Err(Error::invalid("WAV stream must be audio"));
    }
    let channels = s
        .params
        .channels
        .ok_or_else(|| Error::invalid("WAV muxer: missing channels"))?;
    let sample_rate = s
        .params
        .sample_rate
        .ok_or_else(|| Error::invalid("WAV muxer: missing sample rate"))?;
    // Codec-id directs which `wFormatTag` flavour and on-wire shape the
    // muxer writes. A-law / μ-law take the dedicated tag-6/7 paths;
    // every other id falls back to the PCM/IEEE-FLOAT sample-format
    // lookup. Extensible muxing is opt-in via [`WavMuxOptions`] (see
    // [`open_muxer_with`] below) — the default path writes the
    // legacy 16-byte `WAVEFORMAT` for maximum compatibility.
    let shape = wire_shape_for_params(&s.params)?;
    Ok(Box::new(WavMuxer {
        output,
        channels,
        sample_rate,
        shape,
        extensible: None,
        acid: None,
        chna: None,
        bext: None,
        cue: None,
        plst: None,
        adtl: None,
        rf64: Rf64Mode::Never,
        riff_size_offset: 0,
        data_size_offset: 0,
        fact_size_offset: None,
        ds64_body_offset: None,
        magic_is_64bit: false,
        data_bytes: 0,
        header_written: false,
        trailer_written: false,
    }))
}

/// Optional muxer configuration: write a `WAVE_FORMAT_EXTENSIBLE`
/// (`wFormatTag = 0xFFFE`) header with the supplied `dwChannelMask`,
/// `wValidBitsPerSample`, and SubFormat GUID. See
/// `docs/container/riff/waveformatextensible/README.md` §"Channel-mask"
/// for the standard layouts.
///
/// When `valid_bits_per_sample` is `None` the muxer reuses the
/// container `wBitsPerSample` (computed from the codec's
/// `SampleFormat`). When `subformat` is `None` the muxer picks the
/// well-known `KSDATAFORMAT_SUBTYPE_*` GUID for the codec id (PCM /
/// IEEE_FLOAT / ALAW / MULAW).
#[derive(Clone, Debug, Default)]
pub struct WavMuxOptions {
    extensible: Option<ExtensibleOpts>,
    acid: Option<AcidChunk>,
    chna: Option<ChnaChunk>,
    bext: Option<BextChunk>,
    cue: Option<CueChunk>,
    plst: Option<PlaylistChunk>,
    adtl: Option<AdtlChunk>,
    rf64: Rf64Mode,
}

/// How the muxer handles the 64-bit-extended (RF64 / BW64) large-file
/// form (EBU Tech 3306 v2 / ITU-R BS.2088-2 §3–§4).
///
/// A plain `RIFF`/`WAVE` file stores the top-level RIFF size, the
/// `data` chunk size, and the `fact` sample count in 32-bit fields,
/// so it cannot describe a payload larger than 4 GiB − 1. The 64-bit
/// form keeps those 32-bit fields but sets each to the
/// `0xFFFFFFFF` sentinel and carries the real 64-bit values in a
/// mandatory `ds64` chunk placed immediately after the form type
/// (`WAVE`). The top-level magic also changes from `RIFF` to `RF64`
/// (or `BW64` when an ADM `chna` chunk is present — ITU-R BS.2088).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rf64Mode {
    /// Plain 32-bit `RIFF`/`WAVE`. `write_trailer` errors if the
    /// payload would overflow a 32-bit size field. This is the
    /// historical default and keeps short files byte-identical.
    #[default]
    Never,
    /// Reserve a `ds64`-sized `JUNK` placeholder chunk immediately
    /// after `WAVE` (BS.2088-2 §3.6 "File structure with JUNK chunk").
    /// If the finished file fits in 32 bits the placeholder is left as
    /// an inert `JUNK` chunk and the file is a normal `RIFF`/`WAVE`;
    /// if it overflows, the placeholder is promoted in place to a
    /// `ds64` chunk and the magic flips to `RF64`/`BW64` — the
    /// on-the-fly conversion described in BS.2088-2 §4.2. This is the
    /// recording-application pattern: cheap up front, never fails on
    /// overflow.
    Reserve,
    /// Always emit the 64-bit form: a `ds64` chunk after `WAVE`, the
    /// `0xFFFFFFFF` sentinel in the legacy size fields, and the
    /// `RF64`/`BW64` magic — regardless of the final payload size.
    Force,
}

#[derive(Clone, Debug)]
struct ExtensibleOpts {
    channel_mask: u32,
    valid_bits_per_sample: Option<u16>,
    subformat: Option<[u8; 16]>,
}

impl WavMuxOptions {
    /// Opt into `WAVE_FORMAT_EXTENSIBLE` muxing with the supplied
    /// `dwChannelMask`. The muxer derives `wValidBitsPerSample` and
    /// SubFormat from the codec-id automatically; use the
    /// finer-grained setters to override.
    pub fn with_extensible(mut self, channel_mask: u32) -> Self {
        self.extensible = Some(ExtensibleOpts {
            channel_mask,
            valid_bits_per_sample: None,
            subformat: None,
        });
        self
    }

    /// Override `wValidBitsPerSample` for an extensible stream. Has no
    /// effect unless [`Self::with_extensible`] was also called.
    pub fn with_valid_bits_per_sample(mut self, valid_bps: u16) -> Self {
        if let Some(opts) = self.extensible.as_mut() {
            opts.valid_bits_per_sample = Some(valid_bps);
        }
        self
    }

    /// Override the 16-byte SubFormat GUID. Has no effect unless
    /// [`Self::with_extensible`] was also called.
    pub fn with_subformat(mut self, guid: [u8; 16]) -> Self {
        if let Some(opts) = self.extensible.as_mut() {
            opts.subformat = Some(guid);
        }
        self
    }

    /// Emit an Acidizer `acid` metadata chunk (24-byte body, layout
    /// per [`AcidChunk`]) ahead of the `data` chunk.
    pub fn with_acid(mut self, acid: AcidChunk) -> Self {
        self.acid = Some(acid);
        self
    }

    /// Emit a BW64/ADM `chna` (channel-allocation) chunk ahead of the
    /// `data` chunk (layout per [`ChnaChunk`] / ITU-R BS.2088-2 §8.1).
    /// The chunk body is always even-sized, so no pad byte is written.
    pub fn with_chna(mut self, chna: ChnaChunk) -> Self {
        self.chna = Some(chna);
        self
    }

    /// Emit a BWF `bext` (Broadcast Audio Extension) chunk ahead of the
    /// `data` chunk (602-byte fixed struct + optional `CodingHistory`
    /// tail — see [`BextChunk`] / EBU Tech 3285 v2 §2.3). When the body
    /// length is odd (an odd-length `CodingHistory`) the muxer writes the
    /// RIFF word-alignment pad byte after it.
    pub fn with_bext(mut self, bext: BextChunk) -> Self {
        self.bext = Some(bext);
        self
    }

    /// Emit a `cue ` (cue-points) chunk after the `data` waveform
    /// (RIFF MCI §3 "Cue-Points Chunk"). Cue / playlist / associated-
    /// data chunks reference sample positions in the `data` payload, so
    /// they are written in the trailer once the payload is complete —
    /// the conventional placement, and the one a sibling `plst`/`adtl`
    /// expects. The body is `4 + N*24` bytes, always even.
    pub fn with_cue(mut self, cue: CueChunk) -> Self {
        self.cue = Some(cue);
        self
    }

    /// Emit a `plst` (playlist) chunk after the `data` waveform (RIFF
    /// MCI §3 "Playlist Chunk"). Each segment's `cue_id` should match a
    /// [`CuePoint::name`] supplied via [`Self::with_cue`]. The body is
    /// `4 + N*12` bytes, always even.
    pub fn with_plst(mut self, plst: PlaylistChunk) -> Self {
        self.plst = Some(plst);
        self
    }

    /// Emit a `LIST adtl` (Associated Data List) chunk after the `data`
    /// waveform (RIFF MCI §3 "Associated Data Chunk"). Each entry's
    /// `name` should match a [`CuePoint::name`] supplied via
    /// [`Self::with_cue`]. Odd-length `labl`/`note`/`ltxt` sub-chunks
    /// get the RIFF word-alignment pad byte automatically.
    pub fn with_adtl(mut self, adtl: AdtlChunk) -> Self {
        self.adtl = Some(adtl);
        self
    }

    /// Select the 64-bit-extended (RF64 / BW64) large-file behaviour —
    /// see [`Rf64Mode`].
    ///
    /// - [`Rf64Mode::Never`] (default): plain `RIFF`/`WAVE`; the muxer
    ///   errors if the payload would overflow a 32-bit size field.
    /// - [`Rf64Mode::Reserve`]: write a `ds64`-sized `JUNK` placeholder
    ///   up front and promote it to `ds64` + `RF64`/`BW64` only if the
    ///   finished file overflows 32 bits (BS.2088-2 §3.6 / §4.2
    ///   on-the-fly conversion).
    /// - [`Rf64Mode::Force`]: always emit the 64-bit form.
    ///
    /// When a `chna` chunk is also requested (an ADM file), the
    /// promoted / forced magic is `BW64` rather than `RF64` per
    /// ITU-R BS.2088; otherwise it is `RF64` per EBU Tech 3306.
    pub fn with_rf64(mut self, mode: Rf64Mode) -> Self {
        self.rf64 = mode;
        self
    }
}

/// Open the WAV muxer with caller-controlled `WAVEFORMATEXTENSIBLE`
/// options. Identical to `open_muxer` when `opts ==
/// WavMuxOptions::default()`.
pub fn open_muxer_with(
    output: Box<dyn WriteSeek>,
    streams: &[StreamInfo],
    opts: WavMuxOptions,
) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::unsupported("WAV supports exactly one audio stream"));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Audio {
        return Err(Error::invalid("WAV stream must be audio"));
    }
    let channels = s
        .params
        .channels
        .ok_or_else(|| Error::invalid("WAV muxer: missing channels"))?;
    let sample_rate = s
        .params
        .sample_rate
        .ok_or_else(|| Error::invalid("WAV muxer: missing sample rate"))?;
    let shape = wire_shape_for_params(&s.params)?;
    Ok(Box::new(WavMuxer {
        output,
        channels,
        sample_rate,
        shape,
        extensible: opts.extensible,
        acid: opts.acid,
        chna: opts.chna,
        bext: opts.bext,
        cue: opts.cue,
        plst: opts.plst,
        adtl: opts.adtl,
        rf64: opts.rf64,
        riff_size_offset: 0,
        data_size_offset: 0,
        fact_size_offset: None,
        ds64_body_offset: None,
        magic_is_64bit: false,
        data_bytes: 0,
        header_written: false,
        trailer_written: false,
    }))
}

/// On-wire shape the muxer needs to know per codec — distinguishes
/// PCM/IEEE-float (bit-depth driven) from A-law/μ-law (fixed
/// 8 bits / sample).
#[derive(Clone, Copy, Debug)]
enum WireShape {
    /// Integer PCM (`wFormatTag = 0x0001`). bits = `wBitsPerSample`.
    IntPcm { bits: u16 },
    /// IEEE float PCM (`wFormatTag = 0x0003`). bits = 32 or 64.
    FloatPcm { bits: u16 },
    /// G.711 A-law (`wFormatTag = 0x0006`), 8 bits per sample.
    Alaw,
    /// G.711 μ-law (`wFormatTag = 0x0007`), 8 bits per sample.
    Mulaw,
}

impl WireShape {
    fn bits_per_sample(self) -> u16 {
        match self {
            WireShape::IntPcm { bits } | WireShape::FloatPcm { bits } => bits,
            WireShape::Alaw | WireShape::Mulaw => 8,
        }
    }

    fn format_tag(self) -> u16 {
        match self {
            WireShape::IntPcm { .. } => FMT_PCM,
            WireShape::FloatPcm { .. } => FMT_IEEE_FLOAT,
            WireShape::Alaw => FMT_ALAW,
            WireShape::Mulaw => FMT_MULAW,
        }
    }

    fn well_known_guid(self) -> [u8; 16] {
        match self {
            WireShape::IntPcm { .. } => GUID_PCM,
            WireShape::FloatPcm { .. } => GUID_IEEE_FLOAT,
            WireShape::Alaw => GUID_ALAW,
            WireShape::Mulaw => GUID_MULAW,
        }
    }
}

fn wire_shape_for_params(p: &CodecParameters) -> Result<WireShape> {
    match p.codec_id.as_str() {
        "pcm_alaw" => return Ok(WireShape::Alaw),
        "pcm_mulaw" => return Ok(WireShape::Mulaw),
        _ => {}
    }
    let fmt = p
        .sample_format
        .or_else(|| super::pcm::sample_format_for(&p.codec_id))
        .ok_or_else(|| Error::unsupported(format!("WAV: unknown PCM codec {}", p.codec_id)))?;
    Ok(match fmt {
        SampleFormat::U8 => WireShape::IntPcm { bits: 8 },
        SampleFormat::S16 => WireShape::IntPcm { bits: 16 },
        SampleFormat::S24 => WireShape::IntPcm { bits: 24 },
        SampleFormat::S32 => WireShape::IntPcm { bits: 32 },
        SampleFormat::F32 => WireShape::FloatPcm { bits: 32 },
        SampleFormat::F64 => WireShape::FloatPcm { bits: 64 },
        other => {
            return Err(Error::unsupported(format!(
                "WAV muxer cannot write sample format {:?}",
                other
            )));
        }
    })
}

struct WavMuxer {
    output: Box<dyn WriteSeek>,
    channels: u16,
    sample_rate: u32,
    shape: WireShape,
    extensible: Option<ExtensibleOpts>,
    /// Caller-supplied Acidizer metadata, emitted as a 24-byte `acid`
    /// chunk between the format chunks and `data` when present.
    acid: Option<AcidChunk>,
    /// Caller-supplied BW64/ADM channel-allocation metadata, emitted as
    /// a `chna` chunk ahead of `data` when present (ITU-R BS.2088-2
    /// §8.1).
    chna: Option<ChnaChunk>,
    /// Caller-supplied BWF Broadcast Audio Extension metadata, emitted as
    /// a `bext` chunk ahead of `data` when present (EBU Tech 3285 v2
    /// §2.3).
    bext: Option<BextChunk>,
    /// Caller-supplied cue-points, emitted as a `cue ` chunk *after*
    /// `data` in the trailer when present (RIFF MCI §3).
    cue: Option<CueChunk>,
    /// Caller-supplied playlist, emitted as a `plst` chunk after `data`
    /// in the trailer when present (RIFF MCI §3).
    plst: Option<PlaylistChunk>,
    /// Caller-supplied associated-data list, emitted as a `LIST adtl`
    /// chunk after `data` in the trailer when present (RIFF MCI §3).
    adtl: Option<AdtlChunk>,
    /// 64-bit-extended (RF64 / BW64) large-file behaviour — see
    /// [`Rf64Mode`]. Drives whether a `ds64` chunk (or its `JUNK`
    /// placeholder) is reserved after `WAVE` and whether the trailer
    /// promotes the file to the 64-bit form.
    rf64: Rf64Mode,
    riff_size_offset: u64,
    data_size_offset: u64,
    /// File offset of the `ds64`/`JUNK` placeholder chunk *body* (the
    /// first byte after its 8-byte header), set in `write_header` when
    /// `rf64 != Never`. `write_trailer` writes the 64-bit size fields
    /// here (and rewrites the chunk id to `ds64` + the magic to
    /// `RF64`/`BW64`) when promotion is required. `None` for plain
    /// `RIFF`/`WAVE` output.
    ds64_body_offset: Option<u64>,
    /// Set once `write_trailer` has flipped the top-level magic to the
    /// 64-bit form (`Force`, or `Reserve` on overflow). Used so the
    /// RIFF/`ds64` size patching writes the sentinel rather than the
    /// real 32-bit value.
    magic_is_64bit: bool,
    /// Offset of the `dwFileSize` field inside the `fact` chunk we
    /// emit ahead of `data` for non-PCM streams (G.711 A-law / μ-law,
    /// and the EXTENSIBLE escape hatch even when the SubFormat
    /// resolves to PCM — RIFF MCI §3 "FACT Chunk" requires it for any
    /// `wFormatTag != WAVE_FORMAT_PCM`). `None` for PCM streams where
    /// the chunk is optional and we skip emitting it to keep PCM
    /// files byte-identical to the pre-r193 muxer output.
    fact_size_offset: Option<u64>,
    data_bytes: u64,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for WavMuxer {
    fn format_name(&self) -> &str {
        "wav"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("WAV header already written"));
        }
        let bits_per_sample = self.shape.bits_per_sample();
        let block_align = (bits_per_sample / 8) * self.channels;
        let byte_rate = self.sample_rate * block_align as u32;

        // On-wire wFormatTag: caller's extensible opt-in overrides the
        // per-codec default (the underlying shape still drives
        // wBitsPerSample / block_align / byte_rate).
        let format_tag = if self.extensible.is_some() {
            FMT_EXTENSIBLE
        } else {
            self.shape.format_tag()
        };

        // Top-level magic. The write-header always emits `RIFF`; for
        // `Rf64Mode::Force` the trailer rewrites the 4 bytes to
        // `RF64`/`BW64` once it knows whether a `chna` (ADM) chunk made
        // it a BW64 file. For `Rf64Mode::Reserve` the magic is left as
        // `RIFF` unless the trailer detects a 32-bit overflow.
        self.output.write_all(b"RIFF")?;
        self.riff_size_offset = self.output.stream_position()?;
        self.output.write_all(&0u32.to_le_bytes())?; // placeholder
        self.output.write_all(b"WAVE")?;

        // 64-bit-extended large-file support (EBU Tech 3306 v2 / ITU-R
        // BS.2088-2 §3–§4). The `ds64` chunk — or its `JUNK` placeholder
        // for the deferred (`Reserve`) form — is written immediately
        // after the `WAVE` form type and before `fmt `, per BS.2088-2
        // §3.6 ("the <JUNK> placeholder chunk placed before the <fmt >
        // chunk") and §4. Body is the fixed 28-byte (`riffSize` +
        // `dataSize` + `sampleCount`/dummy + `tableLength`) struct with
        // no `ChunkSize64` table entries — this muxer never writes a
        // non-`data` chunk that can itself exceed 4 GiB.
        if self.rf64 != Rf64Mode::Never {
            // `ds64` for the always-on (`Force`) form; `JUNK` for the
            // deferred (`Reserve`) placeholder.
            let id: &[u8; 4] = if self.rf64 == Rf64Mode::Force {
                b"ds64"
            } else {
                b"JUNK"
            };
            self.output.write_all(id)?;
            self.output.write_all(&DS64_FIXED_BODY_LEN.to_le_bytes())?;
            self.ds64_body_offset = Some(self.output.stream_position()?);
            // 28 zero bytes — patched (and, for `Reserve`, possibly
            // promoted) in `write_trailer`.
            self.output
                .write_all(&[0u8; DS64_FIXED_BODY_LEN as usize])?;
        }

        // fmt chunk: 16 bytes for plain WAVEFORMAT, 40 bytes for
        // WAVEFORMATEXTENSIBLE (16 + 2 cbSize + 22 ext).
        let fmt_size: u32 = if self.extensible.is_some() { 40 } else { 16 };
        self.output.write_all(b"fmt ")?;
        self.output.write_all(&fmt_size.to_le_bytes())?;
        self.output.write_all(&format_tag.to_le_bytes())?;
        self.output.write_all(&self.channels.to_le_bytes())?;
        self.output.write_all(&self.sample_rate.to_le_bytes())?;
        self.output.write_all(&byte_rate.to_le_bytes())?;
        self.output.write_all(&block_align.to_le_bytes())?;
        self.output.write_all(&bits_per_sample.to_le_bytes())?;

        if let Some(opts) = &self.extensible {
            // cbSize (22) + 22-byte extension. Layout per
            // docs/container/riff/waveformatextensible/README.md
            // §"Structure layout".
            self.output.write_all(&22u16.to_le_bytes())?;
            let valid = opts.valid_bits_per_sample.unwrap_or(bits_per_sample);
            self.output.write_all(&valid.to_le_bytes())?;
            self.output.write_all(&opts.channel_mask.to_le_bytes())?;
            let guid = opts
                .subformat
                .unwrap_or_else(|| self.shape.well_known_guid());
            self.output.write_all(&guid)?;
        }

        // `fact` chunk: required for any `wFormatTag != WAVE_FORMAT_PCM`
        // (RIFF MCI §3 "FACT Chunk"). We emit it for the
        // EXTENSIBLE-tagged path too even when the SubFormat resolves to
        // PCM, because compliant readers dispatch on the on-wire
        // `wFormatTag` first. The 4-byte `dwFileSize` is patched in
        // `write_trailer` once we know the per-channel sample count.
        if format_tag != FMT_PCM {
            self.output.write_all(b"fact")?;
            self.output.write_all(&4u32.to_le_bytes())?;
            self.fact_size_offset = Some(self.output.stream_position()?);
            self.output.write_all(&0u32.to_le_bytes())?; // placeholder dwFileSize
        }

        // `bext` chunk: caller-supplied BWF Broadcast Audio Extension
        // metadata (EBU Tech 3285 v2 §2.3). The fixed struct is 602
        // bytes; the optional `CodingHistory` tail can make the body
        // odd, so a RIFF word-alignment pad byte is written when needed.
        if let Some(bext) = &self.bext {
            let body = bext.to_bytes();
            self.output.write_all(b"bext")?;
            self.output.write_all(&(body.len() as u32).to_le_bytes())?;
            self.output.write_all(&body)?;
            if body.len() % 2 == 1 {
                self.output.write_all(&[0u8])?;
            }
        }

        // `acid` chunk: caller-supplied Acidizer loop/tempo metadata
        // (24-byte fixed body — see [`AcidChunk`]). Even-sized, so no
        // pad byte is needed.
        if let Some(acid) = &self.acid {
            self.output.write_all(b"acid")?;
            self.output
                .write_all(&(AcidChunk::BODY_LEN as u32).to_le_bytes())?;
            self.output.write_all(&acid.to_bytes())?;
        }

        // `chna` chunk: caller-supplied BW64/ADM channel-allocation
        // records (ITU-R BS.2088-2 §8.1). The body is `4 + N*40` bytes,
        // always even, so no inter-chunk pad byte is needed.
        if let Some(chna) = &self.chna {
            let body = chna.to_bytes();
            self.output.write_all(b"chna")?;
            self.output.write_all(&(body.len() as u32).to_le_bytes())?;
            self.output.write_all(&body)?;
        }

        self.output.write_all(b"data")?;
        self.data_size_offset = self.output.stream_position()?;
        self.output.write_all(&0u32.to_le_bytes())?; // placeholder

        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("WAV muxer: write_header not called"));
        }
        self.output.write_all(&packet.data)?;
        self.data_bytes += packet.data.len() as u64;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        // Pad data chunk to even length.
        if self.data_bytes % 2 == 1 {
            self.output.write_all(&[0u8])?;
        }

        // Trailing metadata chunks (RIFF MCI §3 optional `<other-ck>`
        // that may follow the waveform). These reference sample
        // positions in the `data` payload, so they are emitted here —
        // after `data` is complete — rather than in `write_header`. They
        // are counted in `riff_size` (the cursor advances before `end` is
        // captured below) but never in `data_size`.
        if let Some(cue) = &self.cue {
            let body = cue.to_bytes(); // 4 + N*24, always even
            self.output.write_all(b"cue ")?;
            self.output.write_all(&(body.len() as u32).to_le_bytes())?;
            self.output.write_all(&body)?;
        }
        if let Some(plst) = &self.plst {
            let body = plst.to_bytes(); // 4 + N*12, always even
            self.output.write_all(b"plst")?;
            self.output.write_all(&(body.len() as u32).to_le_bytes())?;
            self.output.write_all(&body)?;
        }
        if let Some(adtl) = &self.adtl {
            // `to_list_body` already includes the `adtl` list type and
            // each sub-chunk's word-alignment pad; the enclosing `LIST`
            // size is the list-body length. The list body is even iff
            // every sub-chunk was padded, which `to_list_body` ensures,
            // so no inter-chunk pad is needed after the LIST.
            let body = adtl.to_list_body();
            self.output.write_all(b"LIST")?;
            self.output.write_all(&(body.len() as u32).to_le_bytes())?;
            self.output.write_all(&body)?;
            if body.len() % 2 == 1 {
                self.output.write_all(&[0u8])?;
            }
        }

        let end = self.output.stream_position()?;

        // True 64-bit RIFF and `data` sizes (the values a `ds64` chunk
        // carries). `riffSize` is the whole file minus the 8-byte
        // `RIFF`/size header (EBU Tech 3306 v2 §3 / ITU-R BS.2088-2 §4.2
        // `bw64Size`); `dataSize` is the un-padded payload length.
        let riff_size = end - 8;
        let data_size = self.data_bytes;

        // Per-channel sample count for the `fact` chunk (and the
        // `ds64.sampleCount`/dummy slot). PCM/G.711/EXTENSIBLE ship
        // pre-framed payload, so `data_bytes / block_align` is exact.
        let bits_per_sample = self.shape.bits_per_sample() as u64;
        let block_align = (bits_per_sample / 8) * self.channels as u64;
        let sample_count = data_size.checked_div(block_align).unwrap_or(0);

        // Decide whether the file must use the 64-bit-extended form.
        // `Force` always does; `Reserve` promotes only on a real 32-bit
        // overflow (BS.2088-2 §4.2 on-the-fly conversion); `Never`
        // errors on overflow as the historical muxer did.
        let overflow = riff_size > u32::MAX as u64 || data_size > u32::MAX as u64;
        let use_64bit = match self.rf64 {
            Rf64Mode::Force => true,
            Rf64Mode::Reserve => overflow,
            Rf64Mode::Never => {
                if overflow {
                    return Err(Error::other(
                        "WAV file exceeds 4 GiB; use WavMuxOptions::with_rf64 for the \
                         RF64/BW64 64-bit form",
                    ));
                }
                false
            }
        };

        if use_64bit {
            // 64-bit (RF64 / BW64) form. The `ds64` placeholder was
            // reserved in `write_header`; here we fill its body, flip a
            // `Reserve`-mode `JUNK` id to `ds64`, set the legacy 32-bit
            // size fields to the `0xFFFFFFFF` sentinel, and rewrite the
            // top-level magic. BW64 when an ADM `chna` chunk is present
            // (ITU-R BS.2088), else RF64 (EBU Tech 3306).
            let ds64_body = self
                .ds64_body_offset
                .ok_or_else(|| Error::other("RF64 muxer: ds64 placeholder not reserved"))?;

            // ds64 body: riffSize(8) + dataSize(8) + sampleCount(8) +
            // tableLength(4) = 28 bytes, no ChunkSize64 entries.
            self.output.seek(SeekFrom::Start(ds64_body))?;
            self.output.write_all(&riff_size.to_le_bytes())?;
            self.output.write_all(&data_size.to_le_bytes())?;
            self.output.write_all(&sample_count.to_le_bytes())?;
            self.output.write_all(&0u32.to_le_bytes())?; // tableLength = 0

            // Promote a `Reserve`-mode `JUNK` placeholder to `ds64`
            // (BS.2088-2 §4.2: "Replace the ckID <JUNK> with <ds64>").
            // The `Force` path already wrote `ds64` in the header.
            if self.rf64 == Rf64Mode::Reserve {
                self.output.seek(SeekFrom::Start(ds64_body - 8))?;
                self.output.write_all(b"ds64")?;
            }

            // Legacy 32-bit size fields carry the sentinel.
            self.output.seek(SeekFrom::Start(self.riff_size_offset))?;
            self.output.write_all(&SIZE64_SENTINEL.to_le_bytes())?;
            self.output.seek(SeekFrom::Start(self.data_size_offset))?;
            self.output.write_all(&SIZE64_SENTINEL.to_le_bytes())?;
            if let Some(off) = self.fact_size_offset {
                self.output.seek(SeekFrom::Start(off))?;
                self.output.write_all(&SIZE64_SENTINEL.to_le_bytes())?;
            }

            // Top-level magic: BW64 for ADM (chna present) else RF64.
            let magic: &[u8; 4] = if self.chna.is_some() {
                b"BW64"
            } else {
                b"RF64"
            };
            self.output.seek(SeekFrom::Start(0))?;
            self.output.write_all(magic)?;
            self.magic_is_64bit = true;
        } else {
            // Plain 32-bit `RIFF`/`WAVE`. Any reserved `JUNK`
            // placeholder is left inert (a valid, ignorable RIFF chunk).
            let data_size_u32 = data_size as u32; // checked non-overflow above
            self.output.seek(SeekFrom::Start(self.data_size_offset))?;
            self.output.write_all(&data_size_u32.to_le_bytes())?;

            if let Some(off) = self.fact_size_offset {
                let sample_count_u32: u32 = sample_count
                    .try_into()
                    .map_err(|_| Error::other("WAV fact sample count exceeds u32"))?;
                self.output.seek(SeekFrom::Start(off))?;
                self.output.write_all(&sample_count_u32.to_le_bytes())?;
            }

            let riff_size_u32 = riff_size as u32; // checked non-overflow above
            self.output.seek(SeekFrom::Start(self.riff_size_offset))?;
            self.output.write_all(&riff_size_u32.to_le_bytes())?;
        }

        self.output.seek(SeekFrom::Start(end))?;
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecParameters, MediaType};

    fn make_stream(fmt: SampleFormat, ch: u16, sr: u32) -> StreamInfo {
        let mut params = CodecParameters::audio(super::super::pcm::codec_id_for(fmt).unwrap());
        params.media_type = MediaType::Audio;
        params.channels = Some(ch);
        params.sample_rate = Some(sr);
        params.sample_format = Some(fmt);
        StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, sr as i64),
            duration: None,
            start_time: Some(0),
            params,
        }
    }

    fn wave_format_tag(p: &CodecParameters) -> Option<u16> {
        match p.tag.as_ref()? {
            oxideav_core::CodecTag::WaveFormat(t) => Some(*t),
            _ => None,
        }
    }

    fn make_g711_stream(codec: &str, ch: u16, sr: u32) -> StreamInfo {
        let mut params = CodecParameters::audio(CodecId::new(codec));
        params.media_type = MediaType::Audio;
        params.channels = Some(ch);
        params.sample_rate = Some(sr);
        // G.711 expands to S16 once decoded — sample_format describes
        // the post-decode shape, matching the round-75 oxideav-avi
        // convention. The on-wire packets are 8-bit codewords.
        params.sample_format = Some(SampleFormat::S16);
        StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, sr as i64),
            duration: None,
            start_time: Some(0),
            params,
        }
    }

    /// Mux a single-packet stream through `open_muxer_with` to a
    /// uniquely-named tmpfile, then return both the encoded bytes and
    /// an open demuxer over them. The tmpfile is removed after the
    /// read. Avoids `Cursor<&mut Vec<u8>>` lifetime traps —
    /// `Box<dyn WriteSeek>` requires `'static`.
    fn mux_to_bytes(
        stream: &StreamInfo,
        payload: &[u8],
        opts: WavMuxOptions,
        tag: &str,
    ) -> Vec<u8> {
        let tmp = std::env::temp_dir().join(format!("oxideav-basic-wav-r77-{tag}.wav"));
        let _ = std::fs::remove_file(&tmp);
        {
            let f = std::fs::File::create(&tmp).unwrap();
            let ws: Box<dyn WriteSeek> = Box::new(f);
            let mut mux = open_muxer_with(ws, std::slice::from_ref(stream), opts).unwrap();
            mux.write_header().unwrap();
            let pkt = Packet::new(0, stream.time_base, payload.to_vec());
            mux.write_packet(&pkt).unwrap();
            mux.write_trailer().unwrap();
        }
        let bytes = std::fs::read(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        bytes
    }

    fn open_demux_from_bytes(bytes: Vec<u8>) -> Box<dyn Demuxer> {
        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
        open_demuxer(rs, &oxideav_core::NullCodecResolver).unwrap()
    }

    #[test]
    fn round_trip_s16_mono() {
        // Write then read back a small S16 mono WAV via the public demuxer/muxer paths.
        let samples: Vec<i16> = (0..1000).map(|i| ((i * 32) - 16000) as i16).collect();
        let mut payload = Vec::with_capacity(samples.len() * 2);
        for s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }

        // Mux to a temp file, then demux and compare.
        let stream = make_stream(SampleFormat::S16, 1, 48_000);
        let tmp = std::env::temp_dir().join("oxideav-basic-wav-test.wav");
        {
            let f = std::fs::File::create(&tmp).unwrap();
            let ws: Box<dyn WriteSeek> = Box::new(f);
            let mut mux = open_muxer(ws, std::slice::from_ref(&stream)).unwrap();
            mux.write_header().unwrap();
            let pkt = Packet::new(0, stream.time_base, payload.clone());
            mux.write_packet(&pkt).unwrap();
            mux.write_trailer().unwrap();
        }
        let rs: Box<dyn ReadSeek> = Box::new(std::fs::File::open(&tmp).unwrap());
        let mut dmx = open_demuxer(rs, &oxideav_core::NullCodecResolver).unwrap();
        assert_eq!(dmx.format_name(), "wav");
        assert_eq!(dmx.streams().len(), 1);
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
        let mut out_bytes = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out_bytes.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out_bytes, payload);
    }

    /// `WAVE_FORMAT_ALAW (0x0006)` — codec_id is `pcm_alaw`, on-wire
    /// stays 8-bit codewords, sample_format hint is S16 (decoded shape).
    #[test]
    fn round_trip_alaw_mono() {
        // Synthetic A-law codewords — A-law is a byte stream, so any
        // distinct-byte sequence exercises the byte-for-byte plumbing.
        let payload: Vec<u8> = (0..=255u8).collect();
        let stream = make_g711_stream("pcm_alaw", 1, 8_000);
        let bytes = mux_to_bytes(&stream, &payload, WavMuxOptions::default(), "alaw-mono");
        let mut dmx = open_demux_from_bytes(bytes);
        assert_eq!(dmx.streams().len(), 1);
        let s = &dmx.streams()[0];
        assert_eq!(s.params.codec_id, CodecId::new("pcm_alaw"));
        assert_eq!(wave_format_tag(&s.params), Some(FMT_ALAW));
        assert_eq!(s.params.sample_format, Some(SampleFormat::S16));
        // bit_rate is the on-wire rate (8 bits/sample * channels * rate),
        // NOT the post-decode S16 rate.
        assert_eq!(s.params.bit_rate, Some(8 * 8_000));
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, payload);
    }

    /// `WAVE_FORMAT_MULAW (0x0007)` — codec_id is `pcm_mulaw`.
    #[test]
    fn round_trip_mulaw_stereo() {
        let payload: Vec<u8> = (0..512u32).map(|i| (i & 0xFF) as u8).collect();
        let stream = make_g711_stream("pcm_mulaw", 2, 8_000);
        let bytes = mux_to_bytes(&stream, &payload, WavMuxOptions::default(), "mulaw-stereo");
        let mut dmx = open_demux_from_bytes(bytes);
        let s = &dmx.streams()[0];
        assert_eq!(s.params.codec_id, CodecId::new("pcm_mulaw"));
        assert_eq!(wave_format_tag(&s.params), Some(FMT_MULAW));
        // stereo G.711 — block_align = 2 bytes (1 byte/channel),
        // byte_rate = 16000, bit_rate = 128 kbps.
        assert_eq!(s.params.bit_rate, Some(16 * 8_000));
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, payload);
    }

    /// `WAVE_FORMAT_EXTENSIBLE (0xFFFE)` end-to-end — muxer emits the
    /// 40-byte fmt chunk with cbSize = 22, demuxer parses the
    /// extension and exposes channel_mask / valid_bits / SubFormat via
    /// both metadata keys AND the typed accessors.
    #[test]
    fn round_trip_extensible_5_1_pcm() {
        // 6-channel 5.1-Microsoft layout (FL FR FC LFE BL BR) per
        // docs/container/riff/waveformatextensible/README.md
        // §"Channel-mask channel ordering". `payload` is one frame of
        // 6 distinct s16 samples so we can verify frame boundary
        // alignment after the round trip.
        const MASK_5_1: u32 = 0x0003F;
        let frame: [i16; 6] = [-100, 200, -300, 400, -500, 600];
        let mut payload = Vec::new();
        for _ in 0..32 {
            for s in &frame {
                payload.extend_from_slice(&s.to_le_bytes());
            }
        }

        let mut stream = make_stream(SampleFormat::S16, 6, 48_000);
        // Build via open_muxer_with so the muxer emits the EXTENSIBLE
        // fmt chunk. open_muxer (the registry default) writes the
        // legacy 16-byte fmt chunk for maximum compatibility.
        stream.params.codec_id = CodecId::new("pcm_s16le");
        let opts = WavMuxOptions::default().with_extensible(MASK_5_1);
        let buf = mux_to_bytes(&stream, &payload, opts, "ext-5-1-pcm");

        // Sanity-check the on-wire fmt chunk size + cbSize.
        // RIFF(4)+size(4)+WAVE(4)+"fmt "(4)+size(4) == 20 bytes
        // before the fmt body; fmt size at offset 16 must be 40.
        assert_eq!(&buf[12..16], b"fmt ");
        let fmt_size = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
        assert_eq!(fmt_size, 40, "EXTENSIBLE fmt chunk must be 40 bytes");
        // wFormatTag at fmt body[0..2] == 0xFFFE
        assert_eq!(u16::from_le_bytes([buf[20], buf[21]]), FMT_EXTENSIBLE);
        // cbSize at fmt body[16..18] == 22
        assert_eq!(u16::from_le_bytes([buf[36], buf[37]]), 22);

        // Demux it back.
        let dmx = open_demux_from_bytes(buf);
        // SubFormat == KSDATAFORMAT_SUBTYPE_PCM resolves the codec id
        // to the legacy pcm_s16le (the round-75 oxideav-avi convention
        // — the GUID identifies the codec, not the bit depth).
        let s = &dmx.streams()[0];
        assert_eq!(s.params.codec_id, CodecId::new("pcm_s16le"));
        assert_eq!(wave_format_tag(&s.params), Some(FMT_EXTENSIBLE));

        // Metadata round-trip.
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:fmt.channel_mask"),
            Some(&format!("0x{MASK_5_1:08X}"))
        );
        assert_eq!(
            md.get("wav:fmt.valid_bits_per_sample"),
            Some(&"16".to_string())
        );
        assert_eq!(
            md.get("wav:fmt.subformat"),
            Some(&"00000001-0000-0010-8000-00AA00389B71".to_string())
        );
        // 0x3F == FRONT_LEFT | FRONT_RIGHT | FRONT_CENTER |
        // LOW_FREQUENCY | BACK_LEFT | BACK_RIGHT (the canonical 5.1
        // layout), decoded LSB-first.
        assert_eq!(
            md.get("wav:fmt.channel_layout"),
            Some(
                &"FRONT_LEFT+FRONT_RIGHT+FRONT_CENTER+LOW_FREQUENCY+BACK_LEFT+BACK_RIGHT"
                    .to_string()
            )
        );
    }

    /// Typed accessors on the concrete `WavDemuxer` carry the same
    /// EXTENSIBLE side-info the metadata keys do.
    #[test]
    fn extensible_accessors_match_metadata() {
        const MASK_STEREO: u32 = 0x00003;
        let payload = vec![0u8; 4 * 100]; // 100 stereo s16 frames

        let mut stream = make_stream(SampleFormat::S16, 2, 44_100);
        stream.params.codec_id = CodecId::new("pcm_s16le");

        let opts = WavMuxOptions::default().with_extensible(MASK_STEREO);
        let bytes = mux_to_bytes(&stream, &payload, opts, "ext-stereo-md");
        let dmx = open_demux_from_bytes(bytes.clone());
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:fmt.channel_mask"),
            Some(&format!("0x{MASK_STEREO:08X}"))
        );
        assert_eq!(
            md.get("wav:fmt.valid_bits_per_sample"),
            Some(&"16".to_string())
        );
        // The metadata key and the typed accessor agree. Open the same
        // bytes through the concrete `WavDemuxer` to reach the typed
        // `channel_layout` / `channel_mask` accessors.
        assert_eq!(
            md.get("wav:fmt.channel_layout"),
            Some(&"FRONT_LEFT+FRONT_RIGHT".to_string())
        );
        let typed = open_wav_demuxer(Box::new(std::io::Cursor::new(bytes))).unwrap();
        assert_eq!(
            typed.channel_layout(),
            Some("FRONT_LEFT+FRONT_RIGHT".to_string())
        );
        assert_eq!(typed.channel_mask(), Some(MASK_STEREO));
    }

    /// `channel_mask_layout` decodes the `SPEAKER_*` bitmap exactly per
    /// `docs/container/riff/waveformatextensible/ms-waveformatextensible.html`
    /// §"dwChannelMask": LSB-first, `+`-joined, with each documented bit
    /// mapping to its flag name (0x1..=0x20000).
    #[test]
    fn channel_mask_layout_decoding() {
        // Mask 0 => no assigned positions.
        assert_eq!(channel_mask_layout(0), None);

        // Single bits across the whole defined range.
        assert_eq!(channel_mask_layout(0x1), Some("FRONT_LEFT".to_string()));
        assert_eq!(channel_mask_layout(0x8), Some("LOW_FREQUENCY".to_string()));
        assert_eq!(
            channel_mask_layout(0x20000),
            Some("TOP_BACK_RIGHT".to_string())
        );

        // Mono (FRONT_CENTER only).
        assert_eq!(channel_mask_layout(0x4), Some("FRONT_CENTER".to_string()));

        // Quad (FL+FR+BL+BR) — note ordering follows bit significance,
        // not the mask literal's textual order.
        assert_eq!(
            channel_mask_layout(0x33),
            Some("FRONT_LEFT+FRONT_RIGHT+BACK_LEFT+BACK_RIGHT".to_string())
        );

        // 7.1 (FL+FR+FC+LFE+BL+BR+SL+SR == 0x63F).
        assert_eq!(
            channel_mask_layout(0x63F),
            Some(
                "FRONT_LEFT+FRONT_RIGHT+FRONT_CENTER+LOW_FREQUENCY\
                 +BACK_LEFT+BACK_RIGHT+SIDE_LEFT+SIDE_RIGHT"
                    .to_string()
            )
        );

        // Bits above the highest defined flag are surfaced verbatim so
        // the round-trip information isn't lost.
        assert_eq!(
            channel_mask_layout(0x40001),
            Some("FRONT_LEFT+UNKNOWN(0x40000)".to_string())
        );
        assert_eq!(
            channel_mask_layout(0x80000000),
            Some("UNKNOWN(0x80000000)".to_string())
        );
    }

    /// EXTENSIBLE muxing of A-law: muxer writes `wFormatTag = 0xFFFE`
    /// with the KSDATAFORMAT_SUBTYPE_ALAW GUID; demuxer resolves
    /// codec_id to `pcm_alaw` via the GUID path.
    #[test]
    fn extensible_alaw_through_guid() {
        let payload: Vec<u8> = (0..=255u8).collect();
        let stream = make_g711_stream("pcm_alaw", 1, 8_000);

        let opts = WavMuxOptions::default().with_extensible(0x00004); // FRONT_CENTER
        let bytes = mux_to_bytes(&stream, &payload, opts, "ext-alaw");
        let mut dmx = open_demux_from_bytes(bytes);
        let s = &dmx.streams()[0];
        // The legacy wFormatTag is preserved (0xFFFE) but the codec_id
        // resolves through the SubFormat GUID — the demuxer must
        // dispatch G.711 even though wFormatTag is the EXTENSIBLE
        // escape hatch.
        assert_eq!(s.params.codec_id, CodecId::new("pcm_alaw"));
        assert_eq!(wave_format_tag(&s.params), Some(FMT_EXTENSIBLE));
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, payload);
    }

    /// EXTENSIBLE with an unknown SubFormat GUID — the demuxer must
    /// synthesise a `wav:guid_<canonical>` codec id so downstream
    /// `make_decoder` lookups fail naming the actual GUID rather than
    /// the opaque `0xFFFE` tag.
    #[test]
    fn extensible_unknown_guid_synthesised_id() {
        // Hand-build a minimal EXTENSIBLE WAV with a bogus GUID, then
        // feed it to the demuxer. We bypass the muxer so we can pick
        // an unknown SubFormat directly.
        let bogus_guid: [u8; 16] = [
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC,
            0xDE, 0xF0,
        ];
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes()); // riff size placeholder
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&40u32.to_le_bytes()); // fmt size
        buf.extend_from_slice(&FMT_EXTENSIBLE.to_le_bytes()); // wFormatTag
        buf.extend_from_slice(&1u16.to_le_bytes()); // channels
        buf.extend_from_slice(&44_100u32.to_le_bytes()); // sample_rate
        buf.extend_from_slice(&88_200u32.to_le_bytes()); // byte_rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block_align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits_per_sample
        buf.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        buf.extend_from_slice(&16u16.to_le_bytes()); // wValidBitsPerSample
        buf.extend_from_slice(&0x00004u32.to_le_bytes()); // dwChannelMask (FC)
        buf.extend_from_slice(&bogus_guid); // SubFormat
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes()); // empty data chunk

        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(buf));
        let dmx_res = open_demuxer(rs, &oxideav_core::NullCodecResolver);
        let dmx = dmx_res.expect("unknown-GUID extensible stream still parses");
        let id = dmx.streams()[0].params.codec_id.as_str().to_string();
        assert!(
            id.starts_with("wav:guid_"),
            "unknown GUID must synthesise wav:guid_<text>, got {id:?}"
        );
        // Canonical text form: lowercase hex-dump in the on-wire
        // little-endian first-three-groups layout.
        assert!(
            id.contains("EFBEADDE-FECA-BEBA"),
            "synthesised id must carry the canonical GUID text, got {id:?}"
        );
    }

    /// Hand-build a minimal EXTENSIBLE WAV carrying `guid` as its
    /// SubFormat, `bits` as both `wBitsPerSample` and the union's
    /// `wValidBitsPerSample`, and an empty `data` chunk. Bypasses the
    /// muxer so the SubFormat GUID can be chosen directly.
    fn extensible_wav_with_guid(guid: &[u8; 16], bits: u16) -> Vec<u8> {
        let block_align = bits / 8;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&40u32.to_le_bytes());
        buf.extend_from_slice(&FMT_EXTENSIBLE.to_le_bytes()); // wFormatTag
        buf.extend_from_slice(&1u16.to_le_bytes()); // channels
        buf.extend_from_slice(&44_100u32.to_le_bytes()); // sample_rate
        buf.extend_from_slice(&(44_100 * block_align as u32).to_le_bytes()); // byte_rate
        buf.extend_from_slice(&block_align.to_le_bytes()); // block_align
        buf.extend_from_slice(&bits.to_le_bytes()); // bits_per_sample
        buf.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        buf.extend_from_slice(&bits.to_le_bytes()); // wValidBitsPerSample
        buf.extend_from_slice(&0x00004u32.to_le_bytes()); // dwChannelMask (FC)
        buf.extend_from_slice(guid); // SubFormat
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// `waveformatex_tag` extracts the embedded legacy `wFormatTag` from
    /// any GUID built by the `KSMedia.h` `DEFINE_WAVEFORMATEX_GUID(x)`
    /// template, and returns `None` for a GUID with a non-matching tail.
    /// Per
    /// docs/container/riff/waveformatextensible/ms-converting-format-tags-and-subformat-guids.md.
    #[test]
    fn waveformatex_tag_extracts_embedded_format_tag() {
        assert_eq!(waveformatex_tag(&GUID_PCM), Some(0x0001));
        assert_eq!(waveformatex_tag(&GUID_IEEE_FLOAT), Some(0x0003));
        assert_eq!(waveformatex_tag(&GUID_ALAW), Some(0x0006));
        assert_eq!(waveformatex_tag(&GUID_MULAW), Some(0x0007));
        // A tag never hand-listed as a constant — MP3 (0x0055) — still
        // resolves through the generic template.
        let mut mp3_guid = GUID_PCM;
        mp3_guid[0] = 0x55;
        mp3_guid[1] = 0x00;
        assert_eq!(waveformatex_tag(&mp3_guid), Some(0x0055));
        // A GUID whose tail does not match the WAVEFORMATEX base is not
        // a template GUID at all.
        let bogus: [u8; 16] = [
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
            0x9B, 0x72, // last byte differs from the 0x71 base
        ];
        assert_eq!(waveformatex_tag(&bogus), None);
    }

    /// An EXTENSIBLE stream whose SubFormat is the IEEE-float template
    /// GUID dispatches identically to the legacy `WAVE_FORMAT_IEEE_FLOAT`
    /// path — i.e. through the shared `codec_for_tag` route, not the four
    /// hand-listed constants. Surfaces `wav:fmt.subformat_tag = 0x0003`.
    #[test]
    fn extensible_template_guid_dispatches_through_legacy_tag() {
        let buf = extensible_wav_with_guid(&GUID_IEEE_FLOAT, 32);
        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(buf));
        let dmx = open_demuxer(rs, &oxideav_core::NullCodecResolver)
            .expect("IEEE-float template GUID resolves");
        let s = &dmx.streams()[0];
        assert_eq!(s.params.codec_id.as_str(), "pcm_f32le");
        let md: std::collections::HashMap<_, _> = dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:fmt.subformat_tag").map(String::as_str),
            Some("0x0003")
        );
    }

    /// A WAVEFORMATEX-template GUID carrying a legacy `wFormatTag` this
    /// crate does not map directly (e.g. MP3, 0x0055) is NOT mistaken for
    /// PCM: it falls through to a synthesised `wav:guid_<text>` id, while
    /// still surfacing `wav:fmt.subformat_tag = 0x0055` so a downstream
    /// tool can identify it.
    #[test]
    fn extensible_template_guid_unmapped_tag_surfaces_tag() {
        let mut mp3_guid = GUID_PCM;
        mp3_guid[0] = 0x55; // 0x0055 == MP3
        let buf = extensible_wav_with_guid(&mp3_guid, 16);
        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(buf));
        let dmx = open_demuxer(rs, &oxideav_core::NullCodecResolver)
            .expect("MP3 template GUID still parses");
        let s = &dmx.streams()[0];
        assert!(
            s.params.codec_id.as_str().starts_with("wav:guid_"),
            "unmapped tag must synthesise wav:guid_<text>, got {:?}",
            s.params.codec_id.as_str()
        );
        let md: std::collections::HashMap<_, _> = dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:fmt.subformat_tag").map(String::as_str),
            Some("0x0055")
        );
    }

    /// `WAVE_FORMAT_EXTENSIBLE` with cbSize < 22 must reject — the spec
    /// mandates a 22-byte extension. Per
    /// docs/container/riff/waveformatextensible/README.md §"Structure
    /// layout": "RIFF `fmt` chunks carrying it are 40 bytes (38 bytes
    /// of struct payload + 2-byte `cbSize = 22`)."
    #[test]
    fn extensible_short_cbsize_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        // 40-byte fmt body but cbSize = 10 (insufficient).
        buf.extend_from_slice(&40u32.to_le_bytes());
        buf.extend_from_slice(&FMT_EXTENSIBLE.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&44_100u32.to_le_bytes());
        buf.extend_from_slice(&88_200u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(&10u16.to_le_bytes()); // cbSize too small
        buf.extend_from_slice(&[0u8; 20]); // padding to reach 40 fmt body bytes
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());

        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(buf));
        let err = open_demuxer(rs, &oxideav_core::NullCodecResolver).err();
        assert!(
            matches!(err, Some(Error::InvalidData(_))),
            "short cbSize must yield Error::InvalidData, got {err:?}"
        );
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `bext`
    /// chunk body inserted between `fmt ` and `data`. Returns the file
    /// bytes ready for `open_demux_from_bytes`. RIFF size is left at 0
    /// (the demuxer doesn't validate it).
    fn wav_with_bext(bext_body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // channels
        buf.extend_from_slice(&8_000u32.to_le_bytes()); // sample_rate
        buf.extend_from_slice(&16_000u32.to_le_bytes()); // byte_rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block_align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits_per_sample
                                                     // bext chunk.
        buf.extend_from_slice(b"bext");
        buf.extend_from_slice(&(bext_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(bext_body);
        if bext_body.len() % 2 == 1 {
            buf.push(0); // RIFF word-alignment pad
        }
        // empty data chunk.
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Assemble a 602-byte BWF v2 `bext` fixed struct plus a coding
    /// history string. Field offsets follow EBU Tech 3285 v2 §2.3.
    #[allow(clippy::too_many_arguments)]
    fn make_bext_v2(
        description: &str,
        originator: &str,
        originator_ref: &str,
        date: &str,
        time: &str,
        time_reference: u64,
        umid: &[u8; 64],
        loudness: [i16; 5],
        coding_history: &str,
    ) -> Vec<u8> {
        fn put_ascii(buf: &mut Vec<u8>, s: &str, width: usize) {
            let bytes = s.as_bytes();
            let n = bytes.len().min(width);
            buf.extend_from_slice(&bytes[..n]);
            buf.resize(buf.len() + (width - n), 0);
        }
        let mut b = Vec::new();
        put_ascii(&mut b, description, 256);
        put_ascii(&mut b, originator, 32);
        put_ascii(&mut b, originator_ref, 32);
        put_ascii(&mut b, date, 10);
        put_ascii(&mut b, time, 8);
        b.extend_from_slice(&(time_reference as u32).to_le_bytes()); // low
        b.extend_from_slice(&((time_reference >> 32) as u32).to_le_bytes()); // high
        b.extend_from_slice(&2u16.to_le_bytes()); // Version = 2
        b.extend_from_slice(umid);
        for v in &loudness {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.resize(BEXT_FIXED_LEN, 0); // Reserved[180] zero-fill to 602
        assert_eq!(b.len(), BEXT_FIXED_LEN);
        b.extend_from_slice(coding_history.as_bytes());
        b
    }

    /// `fmt_loudness` mirrors the EBU Tech 3285 v2 §2.4 round-to-nearest
    /// fixed-point: the stored WORD is `round(100 × value)`, so dividing
    /// by 100 with two decimals recovers the displayed value. The §2.4
    /// negative-number examples (`-22.64` / `-22.65` / `-22.66` stored
    /// as `-2264` / `-2265` / `-2265`) are exercised in reverse here.
    #[test]
    fn bext_loudness_formatting() {
        assert_eq!(fmt_loudness(-2264), "-22.64");
        assert_eq!(fmt_loudness(-2265), "-22.65");
        assert_eq!(fmt_loudness(0), "0.00");
        assert_eq!(fmt_loudness(2300), "23.00");
        assert_eq!(fmt_loudness(-50), "-0.50"); // sub-unity negative keeps the sign
        assert_eq!(fmt_loudness(7), "0.07");
        assert_eq!(fmt_loudness(-1), "-0.01");
    }

    /// Full BWF v2 `bext` round-trip: every field surfaces under its
    /// `wav:bext.*` metadata key, loudness fields decode to two
    /// decimals, the 64-bit TimeReference reassembles, and the UMID is
    /// hex-encoded.
    #[test]
    fn bext_v2_full_metadata() {
        let mut umid = [0u8; 64];
        umid[0] = 0x06;
        umid[1] = 0x0a;
        umid[63] = 0xff;
        let body = make_bext_v2(
            "Scene 1 take 3",
            "OxideAV Recorder",
            "USABC2400001",
            "2026-05-23",
            "14:30:00",
            // 48000 samples/s × 90061 s ≈ a value that spans 32 bits.
            0x0000_0001_2345_6789,
            &umid,
            [-2305, 700, -120, -1850, -2010],
            "A=PCM,F=48000,W=24,M=stereo,T=OxideAV\r\n",
        );
        let bytes = wav_with_bext(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(
            md.get("wav:bext.description"),
            Some(&"Scene 1 take 3".to_string())
        );
        assert_eq!(
            md.get("wav:bext.originator"),
            Some(&"OxideAV Recorder".to_string())
        );
        assert_eq!(
            md.get("wav:bext.originator_reference"),
            Some(&"USABC2400001".to_string())
        );
        assert_eq!(
            md.get("wav:bext.origination_date"),
            Some(&"2026-05-23".to_string())
        );
        assert_eq!(
            md.get("wav:bext.origination_time"),
            Some(&"14:30:00".to_string())
        );
        assert_eq!(
            md.get("wav:bext.time_reference"),
            Some(&0x0000_0001_2345_6789u64.to_string())
        );
        assert_eq!(md.get("wav:bext.version"), Some(&"2".to_string()));
        assert_eq!(
            md.get("wav:bext.loudness_value"),
            Some(&"-23.05".to_string())
        );
        assert_eq!(md.get("wav:bext.loudness_range"), Some(&"7.00".to_string()));
        assert_eq!(
            md.get("wav:bext.max_true_peak_level"),
            Some(&"-1.20".to_string())
        );
        assert_eq!(
            md.get("wav:bext.max_momentary_loudness"),
            Some(&"-18.50".to_string())
        );
        assert_eq!(
            md.get("wav:bext.max_short_term_loudness"),
            Some(&"-20.10".to_string())
        );
        // UMID hex begins with the bytes we set and ends with 0xff.
        let umid_hex = md.get("wav:bext.umid").expect("umid present");
        assert!(umid_hex.starts_with("060a"), "umid hex {umid_hex:?}");
        assert!(umid_hex.ends_with("ff"), "umid hex {umid_hex:?}");
        assert_eq!(umid_hex.len(), 128); // 64 bytes × 2 hex chars
        assert_eq!(
            md.get("wav:bext.coding_history"),
            Some(&"A=PCM,F=48000,W=24,M=stereo,T=OxideAV".to_string())
        );
    }

    /// BWF v0 `bext`: no UMID, no loudness fields. Version = 0 so the
    /// loudness keys must be absent and the (all-zero) UMID suppressed,
    /// while the text + TimeReference fields still surface.
    #[test]
    fn bext_v0_omits_umid_and_loudness() {
        // 602-byte fixed struct, version 0, all UMID/loudness zero.
        let mut body = vec![0u8; BEXT_FIXED_LEN];
        // Description "Field recording".
        let desc = b"Field recording";
        body[..desc.len()].copy_from_slice(desc);
        // TimeReferenceLow = 12345 (offset 338).
        body[338..342].copy_from_slice(&12_345u32.to_le_bytes());
        // Version = 0 (offset 346) — already zero.
        let bytes = wav_with_bext(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(
            md.get("wav:bext.description"),
            Some(&"Field recording".to_string())
        );
        assert_eq!(md.get("wav:bext.version"), Some(&"0".to_string()));
        assert_eq!(
            md.get("wav:bext.time_reference"),
            Some(&"12345".to_string())
        );
        // v0 must not emit loudness or UMID.
        assert!(!md.contains_key("wav:bext.umid"));
        assert!(!md.contains_key("wav:bext.loudness_value"));
        assert!(!md.contains_key("wav:bext.max_true_peak_level"));
    }

    /// A `bext` chunk shorter than the 602-byte fixed struct is
    /// malformed; the parser must skip it without panicking and the
    /// stream must still open (the chunk is treated as opaque).
    #[test]
    fn bext_truncated_is_skipped() {
        let body = vec![0u8; 100]; // < 602
        let bytes = wav_with_bext(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(md.keys().all(|k| !k.starts_with("wav:bext.")));
        // Stream still resolves to PCM s16.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// `BextChunk::{parse,to_bytes}` are exact inverses for a v2 body:
    /// a byte buffer that round-trips through parse→serialize must equal
    /// the original (the Reserved[180] region and the all-zero pad in
    /// each fixed-width string slot are reproduced verbatim).
    #[test]
    fn bext_struct_byte_roundtrip() {
        let mut umid = [0u8; 64];
        umid[0] = 0x06;
        umid[1] = 0x0a;
        umid[63] = 0xff;
        let body = make_bext_v2(
            "Scene 7 take 2",
            "OxideAV Recorder",
            "USABC2400777",
            "2026-06-18",
            "09:15:42",
            0x0000_0001_2345_6789,
            &umid,
            [-2305, 700, -120, -1850, -2010],
            "A=PCM,F=48000,W=24,M=stereo,T=OxideAV\r\n",
        );
        let parsed = BextChunk::parse(&body).expect("parses");
        assert_eq!(parsed.description, "Scene 7 take 2");
        assert_eq!(parsed.version, 2);
        assert_eq!(parsed.time_reference, 0x0000_0001_2345_6789);
        assert_eq!(parsed.umid, umid);
        assert_eq!(parsed.loudness_value, -2305);
        assert_eq!(parsed.max_short_term_loudness, -2010);
        assert_eq!(
            parsed.coding_history,
            "A=PCM,F=48000,W=24,M=stereo,T=OxideAV"
        );
        // The serialized form re-trims the CR/LF tail of the parsed
        // coding-history, so compare against a body rebuilt from the
        // trimmed value — every fixed field and the Reserved[180] region
        // must match byte-for-byte.
        let reser = parsed.to_bytes();
        let mut expected = body.clone();
        // `make_bext_v2` appended the raw CR/LF history; the trimmed
        // round-trip drops the trailing "\r\n", so truncate expected to
        // the trimmed tail for the comparison.
        expected.truncate(BEXT_FIXED_LEN + parsed.coding_history.len());
        assert_eq!(reser, expected);
    }

    /// `WavMuxOptions::with_bext` emits the chunk ahead of `data`, the
    /// demuxer's typed accessor + `wav:bext.*` metadata keys recover the
    /// values, and the PCM payload is untouched. Uses an odd-length
    /// CodingHistory to exercise the RIFF word-alignment pad path.
    #[test]
    fn bext_mux_round_trip() {
        let mut umid = [0u8; 64];
        umid[..4].copy_from_slice(&[0x06, 0x0a, 0x2b, 0x34]);
        let bext = BextChunk {
            description: "Take 5".to_string(),
            originator: "OxideAV".to_string(),
            originator_reference: "REF-001".to_string(),
            origination_date: "2026-06-18".to_string(),
            origination_time: "10:00:00".to_string(),
            time_reference: 0x0000_0002_0000_0001,
            version: 2,
            umid,
            loudness_value: -2264,
            loudness_range: 700,
            max_true_peak_level: -120,
            max_momentary_loudness: -1850,
            max_short_term_loudness: -2010,
            // 13-char (odd) tail → body length 615 (odd) → pad byte.
            coding_history: "A=PCM,F=48000".to_string(),
        };
        let payload: Vec<u8> = (0..400u32).flat_map(|i| (i as i16).to_le_bytes()).collect();
        let stream = make_stream(SampleFormat::S16, 1, 48_000);
        let opts = WavMuxOptions::default().with_bext(bext.clone());
        let bytes = mux_to_bytes(&stream, &payload, opts, "bext-rt");

        // The serialized chunk (header + body) appears verbatim, and
        // because the body is odd-length the next chunk ("data") is
        // word-aligned by a pad byte.
        let body = bext.to_bytes();
        assert_eq!(body.len() % 2, 1, "test intends an odd-length body");
        let mut chunk = b"bext".to_vec();
        chunk.extend_from_slice(&(body.len() as u32).to_le_bytes());
        chunk.extend_from_slice(&body);
        let pos = bytes
            .windows(chunk.len())
            .position(|w| w == &chunk[..])
            .expect("bext chunk present");
        // Byte immediately after the body is the alignment pad (0), then
        // the "data" FOURCC.
        assert_eq!(bytes[pos + chunk.len()], 0);
        assert_eq!(
            &bytes[pos + chunk.len() + 1..pos + chunk.len() + 5],
            b"data"
        );

        let mut dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:bext.description"), Some(&"Take 5".to_string()));
        assert_eq!(
            md.get("wav:bext.time_reference"),
            Some(&0x0000_0002_0000_0001u64.to_string())
        );
        assert_eq!(
            md.get("wav:bext.loudness_value"),
            Some(&"-22.64".to_string())
        );
        assert_eq!(
            md.get("wav:bext.coding_history"),
            Some(&"A=PCM,F=48000".to_string())
        );
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, payload);
    }

    /// Typed `WavDemuxer::bext()` accessor (via the concrete
    /// `open_wav_demuxer` path) returns the parsed struct, and an absent
    /// `bext` chunk yields `None`.
    #[test]
    fn bext_typed_accessor() {
        let mut umid = [0u8; 64];
        umid[0] = 0x42;
        let body = make_bext_v2(
            "Accessor test",
            "Org",
            "Ref",
            "2026-06-18",
            "11:11:11",
            999,
            &umid,
            [-100, 200, -300, 400, -500],
            "A=PCM,F=44100",
        );
        let bytes = wav_with_bext(&body);
        use std::io::Cursor;
        let dmx = open_wav_demuxer(Box::new(Cursor::new(bytes))).unwrap();
        let got = dmx.bext().expect("typed bext present");
        assert_eq!(got.description, "Accessor test");
        assert_eq!(got.version, 2);
        assert_eq!(got.time_reference, 999);
        assert_eq!(got.umid[0], 0x42);
        assert_eq!(got.loudness_value, -100);
        assert_eq!(got.max_short_term_loudness, -500);

        // No `bext` chunk → typed accessor is None.
        let plain = wav_with_smpl_and_inst(None, None);
        let dmx = open_wav_demuxer(Box::new(Cursor::new(plain))).unwrap();
        assert_eq!(dmx.bext(), None);
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `cue `
    /// chunk and optional `LIST adtl` body inserted between `fmt ` and
    /// `data`. Returns the file bytes ready for `open_demux_from_bytes`.
    fn wav_with_cue_and_adtl(cue_body: &[u8], adtl_body: Option<&[u8]>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // cue chunk
        buf.extend_from_slice(b"cue ");
        buf.extend_from_slice(&(cue_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(cue_body);
        if cue_body.len() % 2 == 1 {
            buf.push(0);
        }
        // optional LIST adtl chunk
        if let Some(adtl) = adtl_body {
            // LIST chunk wraps a 4-byte form-type 'adtl' + sub-chunks.
            buf.extend_from_slice(b"LIST");
            buf.extend_from_slice(&((adtl.len() + 4) as u32).to_le_bytes());
            buf.extend_from_slice(b"adtl");
            buf.extend_from_slice(adtl);
            if (adtl.len() + 4) % 2 == 1 {
                buf.push(0);
            }
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Build a single 24-byte `<cue-point>` record per
    /// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3.
    fn cue_point(
        dw_name: u32,
        dw_position: u32,
        fcc_chunk: &[u8; 4],
        dw_chunk_start: u32,
        dw_block_start: u32,
        dw_sample_offset: u32,
    ) -> Vec<u8> {
        let mut b = Vec::with_capacity(24);
        b.extend_from_slice(&dw_name.to_le_bytes());
        b.extend_from_slice(&dw_position.to_le_bytes());
        b.extend_from_slice(fcc_chunk);
        b.extend_from_slice(&dw_chunk_start.to_le_bytes());
        b.extend_from_slice(&dw_block_start.to_le_bytes());
        b.extend_from_slice(&dw_sample_offset.to_le_bytes());
        b
    }

    /// Build a `labl` or `note` sub-chunk body (without the chunk
    /// header) per RIFF MCI §3 "Label and Note Information".
    fn adtl_text_subchunk(id: &[u8; 4], dw_name: u32, text: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(id);
        // Body = 4-byte dwName + ZSTR (text + NUL terminator).
        let body_len = 4 + text.len() + 1;
        b.extend_from_slice(&(body_len as u32).to_le_bytes());
        b.extend_from_slice(&dw_name.to_le_bytes());
        b.extend_from_slice(text.as_bytes());
        b.push(0);
        if body_len % 2 == 1 {
            b.push(0);
        }
        b
    }

    /// Full `cue ` + `LIST adtl` round-trip: two cue points, each with
    /// a `labl` and a `note`, surface under the documented metadata
    /// keys.
    #[test]
    fn cue_and_adtl_full_metadata() {
        // Two cue points (id=1 at sample 0, id=2 at sample 12345).
        let mut cue_body = Vec::new();
        cue_body.extend_from_slice(&2u32.to_le_bytes()); // dwCuePoints
        cue_body.extend(cue_point(1, 0, b"data", 0, 0, 0));
        cue_body.extend(cue_point(2, 12_345, b"data", 0, 0, 12_345));

        // LIST adtl with labl + note for each cue point.
        let mut adtl = Vec::new();
        adtl.extend(adtl_text_subchunk(b"labl", 1, "Intro"));
        adtl.extend(adtl_text_subchunk(b"note", 1, "Fade-in"));
        adtl.extend(adtl_text_subchunk(b"labl", 2, "Verse"));
        adtl.extend(adtl_text_subchunk(b"note", 2, "Vocal entry"));

        let bytes = wav_with_cue_and_adtl(&cue_body, Some(&adtl));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(md.get("wav:cue.count"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:cue.1.position"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:cue.1.fcc_chunk"), Some(&"data".to_string()));
        assert_eq!(md.get("wav:cue.1.chunk_start"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:cue.1.block_start"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:cue.1.sample_offset"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:cue.2.position"), Some(&"12345".to_string()));
        assert_eq!(
            md.get("wav:cue.2.sample_offset"),
            Some(&"12345".to_string())
        );

        assert_eq!(md.get("wav:adtl.labl.1"), Some(&"Intro".to_string()));
        assert_eq!(md.get("wav:adtl.note.1"), Some(&"Fade-in".to_string()));
        assert_eq!(md.get("wav:adtl.labl.2"), Some(&"Verse".to_string()));
        assert_eq!(md.get("wav:adtl.note.2"), Some(&"Vocal entry".to_string()));
    }

    /// `ltxt` sub-chunk surfaces dwSampleLength, FOURCC purpose, and
    /// text under `wav:adtl.ltxt.<dwName>.*`.
    #[test]
    fn adtl_ltxt_segment_metadata() {
        // Single cue point so the ltxt reference is meaningful.
        let mut cue_body = Vec::new();
        cue_body.extend_from_slice(&1u32.to_le_bytes());
        cue_body.extend(cue_point(7, 1000, b"data", 0, 0, 1000));

        // ltxt chunk for cue 7: 4410-sample segment, 'scrp' (script) text.
        let mut ltxt_body = Vec::new();
        ltxt_body.extend_from_slice(&7u32.to_le_bytes()); // dwName
        ltxt_body.extend_from_slice(&4410u32.to_le_bytes()); // dwSampleLength
        ltxt_body.extend_from_slice(b"scrp"); // dwPurpose
        ltxt_body.extend_from_slice(&44u16.to_le_bytes()); // wCountry (United Kingdom)
        ltxt_body.extend_from_slice(&9u16.to_le_bytes()); // wLanguage (English)
        ltxt_body.extend_from_slice(&2u16.to_le_bytes()); // wDialect (UK)
        ltxt_body.extend_from_slice(&1252u16.to_le_bytes()); // wCodePage
        ltxt_body.extend_from_slice(b"Hello world");
        ltxt_body.push(0);

        let mut adtl = Vec::new();
        adtl.extend_from_slice(b"ltxt");
        adtl.extend_from_slice(&(ltxt_body.len() as u32).to_le_bytes());
        adtl.extend_from_slice(&ltxt_body);
        if ltxt_body.len() % 2 == 1 {
            adtl.push(0);
        }

        let bytes = wav_with_cue_and_adtl(&cue_body, Some(&adtl));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(md.get("wav:adtl.ltxt.7.length"), Some(&"4410".to_string()));
        assert_eq!(md.get("wav:adtl.ltxt.7.purpose"), Some(&"scrp".to_string()));
        assert_eq!(
            md.get("wav:adtl.ltxt.7.text"),
            Some(&"Hello world".to_string())
        );
        // §3 locale WORDs: raw decimals plus the Chapter-2 table
        // resolutions shared with CSET.
        assert_eq!(md.get("wav:adtl.ltxt.7.country"), Some(&"44".to_string()));
        assert_eq!(
            md.get("wav:adtl.ltxt.7.country_name"),
            Some(&"United Kingdom".to_string())
        );
        assert_eq!(md.get("wav:adtl.ltxt.7.language"), Some(&"9".to_string()));
        assert_eq!(md.get("wav:adtl.ltxt.7.dialect"), Some(&"2".to_string()));
        assert_eq!(
            md.get("wav:adtl.ltxt.7.language_name"),
            Some(&"UK English".to_string())
        );
        assert_eq!(
            md.get("wav:adtl.ltxt.7.code_page"),
            Some(&"1252".to_string())
        );
    }

    /// `ltxt` locale fields left at zero still surface their raw
    /// decimals (zero = "use the default" per the CSET zero-value
    /// semantics) and resolve to the tables' explicit zero rows.
    #[test]
    fn adtl_ltxt_zero_locale_fields_surface() {
        let mut cue_body = Vec::new();
        cue_body.extend_from_slice(&1u32.to_le_bytes());
        cue_body.extend(cue_point(3, 0, b"data", 0, 0, 0));

        let mut ltxt_body = Vec::new();
        ltxt_body.extend_from_slice(&3u32.to_le_bytes()); // dwName
        ltxt_body.extend_from_slice(&100u32.to_le_bytes()); // dwSampleLength
        ltxt_body.extend_from_slice(b"capt"); // dwPurpose
        ltxt_body.extend_from_slice(&[0u8; 8]); // four zero WORDs

        let mut adtl = Vec::new();
        adtl.extend_from_slice(b"ltxt");
        adtl.extend_from_slice(&(ltxt_body.len() as u32).to_le_bytes());
        adtl.extend_from_slice(&ltxt_body);

        let bytes = wav_with_cue_and_adtl(&cue_body, Some(&adtl));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(md.get("wav:adtl.ltxt.3.country"), Some(&"0".to_string()));
        assert_eq!(
            md.get("wav:adtl.ltxt.3.country_name"),
            Some(&"None".to_string())
        );
        assert_eq!(md.get("wav:adtl.ltxt.3.language"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:adtl.ltxt.3.dialect"), Some(&"0".to_string()));
        assert_eq!(
            md.get("wav:adtl.ltxt.3.language_name"),
            Some(&"None".to_string())
        );
        assert_eq!(md.get("wav:adtl.ltxt.3.code_page"), Some(&"0".to_string()));
        // No text payload past the 20-byte fixed header → no text key.
        assert_eq!(md.get("wav:adtl.ltxt.3.text"), None);
    }

    /// `file` sub-chunk (§3 "Embedded File Information") surfaces the
    /// media type FOURCC and the embedded payload length under
    /// `wav:adtl.file.<dwName>.*` without exposing the payload bytes.
    #[test]
    fn adtl_file_subchunk_metadata() {
        let mut cue_body = Vec::new();
        cue_body.extend_from_slice(&1u32.to_le_bytes());
        cue_body.extend(cue_point(5, 0, b"data", 0, 0, 0));

        // file chunk for cue 5: an embedded 'RDIB' form of 11 bytes
        // (the spec's own example of an embeddable RIFF form type).
        let mut file_body = Vec::new();
        file_body.extend_from_slice(&5u32.to_le_bytes()); // dwName
        file_body.extend_from_slice(b"RDIB"); // dwMedType
        file_body.extend_from_slice(&[0xAAu8; 11]); // fileData

        let mut adtl = Vec::new();
        adtl.extend_from_slice(b"file");
        adtl.extend_from_slice(&(file_body.len() as u32).to_le_bytes());
        adtl.extend_from_slice(&file_body);
        if file_body.len() % 2 == 1 {
            adtl.push(0);
        }
        // A second file chunk (dwName 6) with the spec-allowed zero
        // dwMedType and no fileData payload.
        let mut file2_body = Vec::new();
        file2_body.extend_from_slice(&6u32.to_le_bytes()); // dwName
        file2_body.extend_from_slice(&0u32.to_le_bytes()); // dwMedType = 0
        adtl.extend_from_slice(b"file");
        adtl.extend_from_slice(&(file2_body.len() as u32).to_le_bytes());
        adtl.extend_from_slice(&file2_body);

        let bytes = wav_with_cue_and_adtl(&cue_body, Some(&adtl));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(
            md.get("wav:adtl.file.5.med_type"),
            Some(&"RDIB".to_string())
        );
        assert_eq!(md.get("wav:adtl.file.5.body_len"), Some(&"11".to_string()));
        // Zero med_type renders as plain "0"; empty fileData → 0 len.
        assert_eq!(md.get("wav:adtl.file.6.med_type"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:adtl.file.6.body_len"), Some(&"0".to_string()));
    }

    /// A `file` sub-chunk shorter than its 8-byte fixed header is
    /// skipped as opaque — no keys, no panic.
    #[test]
    fn adtl_file_truncated_is_skipped() {
        let cue_body = 0u32.to_le_bytes().to_vec();
        let mut adtl = Vec::new();
        adtl.extend_from_slice(b"file");
        adtl.extend_from_slice(&6u32.to_le_bytes()); // 6 < 8-byte header
        adtl.extend_from_slice(&[0u8; 6]);
        let bytes = wav_with_cue_and_adtl(&cue_body, Some(&adtl));
        let dmx = open_demux_from_bytes(bytes);
        assert!(dmx
            .metadata()
            .iter()
            .all(|(k, _)| !k.starts_with("wav:adtl.file.")));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `cue ` chunk whose `dwCuePoints` count exceeds the body length
    /// must not panic — the parser surfaces only the records that
    /// actually fit in the body.
    #[test]
    fn cue_truncated_count_is_clamped() {
        // Claim 5 points, ship 1.
        let mut cue_body = Vec::new();
        cue_body.extend_from_slice(&5u32.to_le_bytes());
        cue_body.extend(cue_point(42, 100, b"data", 0, 0, 100));
        let bytes = wav_with_cue_and_adtl(&cue_body, None);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cue.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:cue.42.position"), Some(&"100".to_string()));
        // Stream still opens cleanly.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// An `adtl` list without a matching `cue ` chunk still emits its
    /// `wav:adtl.*` keys — the spec doesn't require the cue chunk to
    /// precede the adtl list, and downstream consumers can cross-
    /// reference dwName values themselves.
    #[test]
    fn adtl_without_cue_still_surfaces() {
        let mut adtl = Vec::new();
        adtl.extend(adtl_text_subchunk(b"labl", 99, "Orphan label"));
        // cue body with zero points exercises the count=0 path.
        let cue_body = 0u32.to_le_bytes().to_vec();
        let bytes = wav_with_cue_and_adtl(&cue_body, Some(&adtl));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cue.count"), Some(&"0".to_string()));
        assert_eq!(
            md.get("wav:adtl.labl.99"),
            Some(&"Orphan label".to_string())
        );
    }

    /// Build a minimal valid PCM WAV with caller-supplied raw `smpl`
    /// and/or `inst` chunks inserted between `fmt ` and `data`. Mirrors
    /// `wav_with_cue_and_adtl` but for the sampler/instrument chunk
    /// pair.
    fn wav_with_smpl_and_inst(smpl_body: Option<&[u8]>, inst_body: Option<&[u8]>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        if let Some(smpl) = smpl_body {
            buf.extend_from_slice(b"smpl");
            buf.extend_from_slice(&(smpl.len() as u32).to_le_bytes());
            buf.extend_from_slice(smpl);
            if smpl.len() % 2 == 1 {
                buf.push(0);
            }
        }
        if let Some(inst) = inst_body {
            buf.extend_from_slice(b"inst");
            buf.extend_from_slice(&(inst.len() as u32).to_le_bytes());
            buf.extend_from_slice(inst);
            if inst.len() % 2 == 1 {
                buf.push(0);
            }
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Build a `smpl` fixed header (36 bytes) followed by N sample-loop
    /// records (24 bytes each).
    #[allow(clippy::too_many_arguments)]
    fn smpl_body(
        manufacturer: u32,
        product: u32,
        sample_period: u32,
        midi_unity_note: u32,
        midi_pitch_fraction: u32,
        smpte_format: u32,
        smpte_offset: u32,
        c_sample_loops_claimed: u32,
        cb_sampler_data: u32,
        loops: &[(u32, u32, u32, u32, u32, u32)],
    ) -> Vec<u8> {
        let mut b = Vec::new();
        for v in [
            manufacturer,
            product,
            sample_period,
            midi_unity_note,
            midi_pitch_fraction,
            smpte_format,
            smpte_offset,
            c_sample_loops_claimed,
            cb_sampler_data,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for &(id, ty, start, end, frac, count) in loops {
            for v in [id, ty, start, end, frac, count] {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b
    }

    /// Full `smpl` round-trip: one loop, every fixed-header field and
    /// the per-loop record surface under the documented metadata keys.
    /// SMPTE offset is decoded as `HH:MM:SS:FF`.
    #[test]
    fn smpl_full_metadata() {
        // SMPTE offset 0x01020304 → 01:02:03:04 (HH MM SS FF).
        let body = smpl_body(
            0x1234,                   // manufacturer
            0xDEAD_BEEF,              // product
            22_675,                   // sample period (ns; ≈ 44.1 kHz)
            60,                       // MIDI middle-C
            0x8000_0000,              // MIDI pitch fraction (½ semitone)
            30,                       // SMPTE 30 fps
            0x01_02_03_04,            // SMPTE offset HH:MM:SS:FF
            1,                        // one sample loop
            0,                        // no sampler-specific data
            &[(7, 0, 0, 1000, 0, 0)], // cue id 7, fwd loop, 0..1000, infinite
        );
        let bytes = wav_with_smpl_and_inst(Some(&body), None);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:smpl.manufacturer"), Some(&"4660".to_string()));
        assert_eq!(md.get("wav:smpl.product"), Some(&"3735928559".to_string()));
        assert_eq!(md.get("wav:smpl.sample_period"), Some(&"22675".to_string()));
        assert_eq!(md.get("wav:smpl.midi_unity_note"), Some(&"60".to_string()));
        assert_eq!(
            md.get("wav:smpl.midi_pitch_fraction"),
            Some(&"2147483648".to_string())
        );
        assert_eq!(md.get("wav:smpl.smpte_format"), Some(&"30".to_string()));
        assert_eq!(
            md.get("wav:smpl.smpte_offset"),
            Some(&"01:02:03:04".to_string())
        );
        assert_eq!(md.get("wav:smpl.sampler_data_len"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:smpl.num_sample_loops"), Some(&"1".to_string()));
        assert_eq!(
            md.get("wav:smpl.loop.0.cue_point_id"),
            Some(&"7".to_string())
        );
        assert_eq!(md.get("wav:smpl.loop.0.type"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:smpl.loop.0.start"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:smpl.loop.0.end"), Some(&"1000".to_string()));
        assert_eq!(md.get("wav:smpl.loop.0.fraction"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:smpl.loop.0.play_count"), Some(&"0".to_string()));
        // Stream still opens cleanly.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `smpl` chunk whose `cSampleLoops` count exceeds the records
    /// that actually fit in the chunk body is clamped to the records
    /// the body carries — defensive vs. writers that lie about the
    /// count (mirrors the `cue ` chunk's clamping behaviour).
    #[test]
    fn smpl_loop_count_clamped_to_body() {
        // Claim 5 loops but provide only 1 — parser must surface
        // num_sample_loops=1.
        let body = smpl_body(
            0,
            0,
            0,
            60,
            0,
            0,
            0,
            /* claim */ 5,
            0,
            &[(1, 0, 0, 100, 0, 0)],
        );
        let bytes = wav_with_smpl_and_inst(Some(&body), None);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:smpl.num_sample_loops"), Some(&"1".to_string()));
        // Only loop.0 should be present; loop.1.. must not be emitted.
        assert!(md.contains_key("wav:smpl.loop.0.cue_point_id"));
        assert!(!md.contains_key("wav:smpl.loop.1.cue_point_id"));
        assert!(!md.contains_key("wav:smpl.loop.4.cue_point_id"));
    }

    /// A `smpl` chunk shorter than the 36-byte fixed struct is
    /// malformed; the parser must skip it without panicking and the
    /// stream must still open.
    #[test]
    fn smpl_truncated_is_skipped() {
        let body = vec![0u8; 20]; // < 36
        let bytes = wav_with_smpl_and_inst(Some(&body), None);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(md.keys().all(|k| !k.starts_with("wav:smpl.")));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Full `inst` round-trip: signed `FineTune` / `Gain` are decoded
    /// as i8 (so `-1` shows as `-1`, not `255`), MIDI note fields are
    /// unsigned.
    #[test]
    fn inst_full_metadata() {
        // FineTune = -3 cents (0xFD), Gain = -6 dB (0xFA).
        let body: Vec<u8> = vec![60, 0xFD, 0xFA, 36, 96, 1, 127];
        let bytes = wav_with_smpl_and_inst(None, Some(&body));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:inst.unshifted_note"), Some(&"60".to_string()));
        assert_eq!(md.get("wav:inst.fine_tune"), Some(&"-3".to_string()));
        assert_eq!(md.get("wav:inst.gain"), Some(&"-6".to_string()));
        assert_eq!(md.get("wav:inst.low_note"), Some(&"36".to_string()));
        assert_eq!(md.get("wav:inst.high_note"), Some(&"96".to_string()));
        assert_eq!(md.get("wav:inst.low_velocity"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:inst.high_velocity"), Some(&"127".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// An `inst` chunk shorter than the 7-byte fixed struct is skipped
    /// as opaque and the stream still resolves.
    #[test]
    fn inst_truncated_is_skipped() {
        let body = vec![0u8; 5]; // < 7
        let bytes = wav_with_smpl_and_inst(None, Some(&body));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(md.keys().all(|k| !k.starts_with("wav:inst.")));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Both `smpl` and `inst` chunks present in the same file surface
    /// under their respective key namespaces without colliding. The
    /// odd-length `inst` chunk forces the 1-byte word-pad path; the
    /// `data` chunk that follows must still be located.
    #[test]
    fn smpl_and_inst_coexist_with_padding() {
        let smpl = smpl_body(0, 0, 0, 64, 0, 0, 0, 0, 0, &[]);
        let inst: Vec<u8> = vec![64, 0, 0, 0, 127, 1, 127]; // 7 bytes → odd
        let bytes = wav_with_smpl_and_inst(Some(&smpl), Some(&inst));
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:smpl.midi_unity_note"), Some(&"64".to_string()));
        assert_eq!(md.get("wav:inst.unshifted_note"), Some(&"64".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `acid`
    /// chunk inserted between `fmt ` and `data`. Mirrors
    /// `wav_with_smpl_and_inst`.
    fn wav_with_acid(acid_body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"acid");
        buf.extend_from_slice(&(acid_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(acid_body);
        if acid_body.len() % 2 == 1 {
            buf.push(0);
        }
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// `AcidChunk::to_bytes` is pinned byte-for-byte against the
    /// documented little-endian layout (flags @0, root note @4,
    /// reserved @6, beats @12, meter @16, tempo @20) and
    /// `AcidChunk::parse` inverts it exactly.
    #[test]
    fn acid_chunk_byte_layout_pinned() {
        let acid = AcidChunk {
            flags: ACID_FLAG_ROOT_NOTE_SET | ACID_FLAG_STRETCH,
            root_note: 57, // A
            reserved: [0x80, 0x00, 0x01, 0x02, 0x03, 0x04],
            num_beats: 16,
            meter: 4,
            tempo: 120.5,
        };
        let bytes = acid.to_bytes();
        #[rustfmt::skip]
        let expected: [u8; 24] = [
            0x06, 0x00, 0x00, 0x00,             // flags = 0x00000006
            0x39, 0x00,                         // root note = 57
            0x80, 0x00, 0x01, 0x02, 0x03, 0x04, // reserved, verbatim
            0x10, 0x00, 0x00, 0x00,             // beats = 16
            0x04, 0x00, 0x00, 0x00,             // meter = 4
            0x00, 0x00, 0xF1, 0x42,             // 120.5f32 LE
        ];
        assert_eq!(bytes, expected);
        assert_eq!(AcidChunk::parse(&bytes), Some(acid));
    }

    /// Full `acid` read path: every documented field surfaces under the
    /// `wav:acid.*` metadata keys, flag bits decode per the staged
    /// Acidizer table, and the root-note name table resolves.
    #[test]
    fn acid_full_metadata() {
        let acid = AcidChunk {
            flags: ACID_FLAG_ONE_SHOT | ACID_FLAG_ROOT_NOTE_SET | ACID_FLAG_HIGH_OCTAVE,
            root_note: 60, // High C
            reserved: [0; 6],
            num_beats: 32,
            meter: 4,
            tempo: 95.0,
        };
        let bytes = wav_with_acid(&acid.to_bytes());
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:acid.flags"), Some(&"0x00000013".to_string()));
        assert_eq!(md.get("wav:acid.one_shot"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:acid.root_note_set"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:acid.stretch"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:acid.disk_based"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:acid.high_octave"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:acid.root_note"), Some(&"60".to_string()));
        assert_eq!(
            md.get("wav:acid.root_note_name"),
            Some(&"High C".to_string())
        );
        assert_eq!(md.get("wav:acid.num_beats"), Some(&"32".to_string()));
        assert_eq!(md.get("wav:acid.meter"), Some(&"4".to_string()));
        assert_eq!(md.get("wav:acid.tempo"), Some(&"95".to_string()));
        // All-zero reserved bytes → no reserved key, exact-size body →
        // no body_len key.
        assert_eq!(md.get("wav:acid.reserved"), None);
        assert_eq!(md.get("wav:acid.body_len"), None);
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Out-of-table root note (not in 48..=71) surfaces the raw value
    /// but no name; nonzero reserved bytes and oversize bodies surface
    /// their observability keys.
    #[test]
    fn acid_out_of_table_root_note_and_extras() {
        let acid = AcidChunk {
            flags: 0,
            root_note: 0,
            reserved: [0xAA, 0, 0, 0, 0, 0xBB],
            num_beats: 8,
            meter: 4,
            tempo: 133.25,
        };
        assert_eq!(acid.root_note_name(), None);
        let mut body = acid.to_bytes().to_vec();
        body.extend_from_slice(&[0xEE, 0xFF]); // future-extension bytes
        let bytes = wav_with_acid(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:acid.root_note"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:acid.root_note_name"), None);
        assert_eq!(md.get("wav:acid.tempo"), Some(&"133.25".to_string()));
        assert_eq!(
            md.get("wav:acid.reserved"),
            Some(&"AA00000000BB".to_string())
        );
        assert_eq!(md.get("wav:acid.body_len"), Some(&"26".to_string()));
    }

    /// A body shorter than the 24-byte fixed struct is opaque-skipped:
    /// no `wav:acid.*` keys, stream still opens.
    #[test]
    fn acid_truncated_is_skipped() {
        let bytes = wav_with_acid(&[0u8; 23]);
        let dmx = open_demux_from_bytes(bytes);
        assert!(!dmx
            .metadata()
            .iter()
            .any(|(k, _)| k.starts_with("wav:acid")));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Write→read round-trip through the public muxer/demuxer paths:
    /// `WavMuxOptions::with_acid` emits the chunk, the demuxer's typed
    /// accessor and metadata keys return the identical values, and the
    /// PCM payload is untouched.
    #[test]
    fn acid_round_trip() {
        let acid = AcidChunk {
            flags: ACID_FLAG_ROOT_NOTE_SET,
            root_note: 50, // D
            reserved: [0; 6],
            num_beats: 64,
            meter: 4,
            tempo: 174.0,
        };
        let payload: Vec<u8> = (0..400u32).flat_map(|i| (i as i16).to_le_bytes()).collect();
        let stream = make_stream(SampleFormat::S16, 1, 44_100);
        let opts = WavMuxOptions::default().with_acid(acid);
        let bytes = mux_to_bytes(&stream, &payload, opts, "acid-rt");
        // The serialized chunk (header + 24-byte body) appears verbatim.
        let mut chunk = b"acid".to_vec();
        chunk.extend_from_slice(&24u32.to_le_bytes());
        chunk.extend_from_slice(&acid.to_bytes());
        assert!(bytes.windows(chunk.len()).any(|w| w == &chunk[..]));

        let mut dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:acid.root_note"), Some(&"50".to_string()));
        assert_eq!(md.get("wav:acid.root_note_name"), Some(&"D".to_string()));
        assert_eq!(md.get("wav:acid.num_beats"), Some(&"64".to_string()));
        assert_eq!(md.get("wav:acid.tempo"), Some(&"174".to_string()));
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, payload);
    }

    /// Typed `WavDemuxer::acid()` accessor (via the concrete
    /// `open_wav_demuxer` path) returns the parsed struct field-for-
    /// field, including the verbatim reserved bytes; absent chunk →
    /// `None`.
    #[test]
    fn acid_typed_accessor() {
        let acid = AcidChunk {
            flags: ACID_FLAG_ONE_SHOT | ACID_FLAG_DISK_BASED,
            root_note: 71, // High B — last entry of the table
            reserved: [1, 2, 3, 4, 5, 6],
            num_beats: 4,
            meter: 3,
            tempo: 60.0,
        };
        let bytes = wav_with_acid(&acid.to_bytes());
        use std::io::Cursor;
        let dmx = open_wav_demuxer(Box::new(Cursor::new(bytes))).unwrap();
        assert_eq!(dmx.acid(), Some(&acid));
        let got = dmx.acid().unwrap();
        assert!(got.one_shot() && got.disk_based());
        assert!(!got.root_note_set() && !got.stretch() && !got.high_octave());
        assert_eq!(got.root_note_name(), Some("High B"));
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:acid.reserved"),
            Some(&"010203040506".to_string())
        );

        // No `acid` chunk → typed accessor is None.
        let plain = wav_with_smpl_and_inst(None, None);
        let dmx = open_wav_demuxer(Box::new(Cursor::new(plain))).unwrap();
        assert_eq!(dmx.acid(), None);
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `chna`
    /// chunk inserted between `fmt ` and `data`. Mirrors
    /// `wav_with_acid`.
    fn wav_with_chna(chna_body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&48_000u32.to_le_bytes());
        buf.extend_from_slice(&192_000u32.to_le_bytes());
        buf.extend_from_slice(&4u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"chna");
        buf.extend_from_slice(&(chna_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(chna_body);
        if chna_body.len() % 2 == 1 {
            buf.push(0);
        }
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Construct an `audioID` record from text fields (right-padding the
    /// fixed-width char arrays with NUL the way an ADM writer does).
    fn audio_id(track_index: u16, uid: &str, track_ref: &str, pack_ref: &str) -> AudioId {
        fn pad<const N: usize>(s: &str) -> [u8; N] {
            let mut out = [0u8; N];
            let b = s.as_bytes();
            out[..b.len()].copy_from_slice(b);
            out
        }
        AudioId {
            track_index,
            uid: pad::<12>(uid),
            track_ref: pad::<14>(track_ref),
            pack_ref: pad::<11>(pack_ref),
            pad: 0,
        }
    }

    /// `ChnaChunk::to_bytes` is pinned byte-for-byte against the
    /// BS.2088-2 §8.3.1 stereo worked example (numTracks=2, numUIDs=2,
    /// two 40-byte `audioID` records, ckSize=84) and `ChnaChunk::parse`
    /// inverts it exactly.
    #[test]
    fn chna_chunk_byte_layout_pinned() {
        let chna = ChnaChunk {
            num_tracks: 2,
            num_uids: 2,
            ids: vec![
                audio_id(1, "ATU_00000001", "AT_00010001_01", "AP_00010002"),
                audio_id(2, "ATU_00000002", "AT_00010002_01", "AP_00010002"),
            ],
        };
        let bytes = chna.to_bytes();
        // ckSize = 4 + N*40 = 4 + 2*40 = 84 (§2 worked example).
        assert_eq!(bytes.len(), 84);
        assert_eq!(chna.body_len(), 84);
        // Pre-amble: numTracks=2, numUIDs=2 (LE WORDs).
        assert_eq!(&bytes[0..4], &[0x02, 0x00, 0x02, 0x00]);
        // First record: trackIndex=1 then the three fixed-width refs.
        assert_eq!(&bytes[4..6], &[0x01, 0x00]);
        assert_eq!(&bytes[6..18], b"ATU_00000001");
        assert_eq!(&bytes[18..32], b"AT_00010001_01");
        assert_eq!(&bytes[32..43], b"AP_00010002");
        assert_eq!(bytes[43], 0); // pad
        assert_eq!(ChnaChunk::parse(&bytes), Some(chna));
    }

    /// Full `chna` read path: counts and every defined record's fields
    /// surface under the `wav:chna.*` keys; PCM stream still resolves.
    #[test]
    fn chna_full_metadata() {
        let chna = ChnaChunk {
            num_tracks: 2,
            num_uids: 2,
            ids: vec![
                audio_id(1, "ATU_00000001", "AT_00010001_01", "AP_00010002"),
                // Linear-PCM essence references an audioChannelFormat
                // (`AC_..._00`) and carries no pack (11 NULs).
                audio_id(2, "ATU_00000002", "AC_00010002_00", ""),
            ],
        };
        let bytes = wav_with_chna(&chna.to_bytes());
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:chna.num_tracks"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:chna.num_uids"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:chna.record_count"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:chna.defined_count"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:chna.0.track_index"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:chna.0.uid"), Some(&"ATU_00000001".to_string()));
        assert_eq!(
            md.get("wav:chna.0.track_ref"),
            Some(&"AT_00010001_01".to_string())
        );
        assert_eq!(
            md.get("wav:chna.0.pack_ref"),
            Some(&"AP_00010002".to_string())
        );
        assert_eq!(md.get("wav:chna.1.track_index"), Some(&"2".to_string()));
        assert_eq!(
            md.get("wav:chna.1.track_ref"),
            Some(&"AC_00010002_00".to_string())
        );
        // No pack on record 1 (all-NUL pack_ref) → key omitted.
        assert_eq!(md.get("wav:chna.1.pack_ref"), None);
        // Exact-size body → no body_len key.
        assert_eq!(md.get("wav:chna.body_len"), None);
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Over-provisioned chunk (§1.3): N=4 records but only 2 are
    /// defined; spare (`track_index == 0`) records are counted but not
    /// surfaced individually, and round-trip is byte-lossless.
    #[test]
    fn chna_over_provisioned_spares() {
        let spare = AudioId {
            track_index: 0,
            uid: [0; 12],
            track_ref: [0; 14],
            pack_ref: [0; 11],
            pad: 0,
        };
        let chna = ChnaChunk {
            num_tracks: 2,
            num_uids: 2,
            ids: vec![
                audio_id(1, "ATU_00000001", "AT_00010001_01", "AP_00010002"),
                audio_id(2, "ATU_00000002", "AT_00010002_01", "AP_00010002"),
                spare,
                spare,
            ],
        };
        // N = (ckSize - 4) / 40 = 4.
        assert_eq!(chna.body_len(), 4 + 4 * 40);
        let bytes = wav_with_chna(&chna.to_bytes());
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:chna.record_count"), Some(&"4".to_string()));
        assert_eq!(md.get("wav:chna.defined_count"), Some(&"2".to_string()));
        // Only the two defined records get per-record keys.
        assert_eq!(md.get("wav:chna.1.track_index"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:chna.2.track_index"), None);
        // Round-trip through parse keeps the spares verbatim.
        assert_eq!(ChnaChunk::parse(&chna.to_bytes()), Some(chna));
    }

    /// A body shorter than the 4-byte count pre-amble is opaque-skipped:
    /// no `wav:chna.*` keys, stream still opens.
    #[test]
    fn chna_truncated_is_skipped() {
        let bytes = wav_with_chna(&[0u8; 3]);
        let dmx = open_demux_from_bytes(bytes);
        assert!(!dmx
            .metadata()
            .iter()
            .any(|(k, _)| k.starts_with("wav:chna")));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A body with trailing bytes that don't fill a whole 40-byte
    /// record: the partial record is ignored and `wav:chna.body_len`
    /// surfaces the raw on-wire length for observability.
    #[test]
    fn chna_trailing_partial_record_surfaces_body_len() {
        let chna = ChnaChunk {
            num_tracks: 1,
            num_uids: 1,
            ids: vec![audio_id(1, "ATU_00000001", "AT_00010001_01", "AP_00010002")],
        };
        let mut body = chna.to_bytes();
        body.extend_from_slice(&[0xEE, 0xFF]); // < 40, ignored
        let bytes = wav_with_chna(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:chna.record_count"), Some(&"1".to_string()));
        // body_len = 4 + 40 + 2 = 46.
        assert_eq!(md.get("wav:chna.body_len"), Some(&"46".to_string()));
    }

    /// Write→read round-trip through the public muxer/demuxer paths:
    /// `WavMuxOptions::with_chna` emits the chunk, the demuxer's typed
    /// accessor and metadata keys return identical values, PCM is
    /// untouched.
    #[test]
    fn chna_round_trip() {
        let chna = ChnaChunk {
            num_tracks: 1,
            num_uids: 1,
            ids: vec![audio_id(1, "ATU_0000000A", "AT_0001000A_01", "AP_0001000B")],
        };
        let payload: Vec<u8> = (0..400u32).flat_map(|i| (i as i16).to_le_bytes()).collect();
        let stream = make_stream(SampleFormat::S16, 1, 44_100);
        let opts = WavMuxOptions::default().with_chna(chna.clone());
        let bytes = mux_to_bytes(&stream, &payload, opts, "chna-rt");
        // The serialized chunk (header + body) appears verbatim.
        let body = chna.to_bytes();
        let mut chunk = b"chna".to_vec();
        chunk.extend_from_slice(&(body.len() as u32).to_le_bytes());
        chunk.extend_from_slice(&body);
        assert!(bytes.windows(chunk.len()).any(|w| w == &chunk[..]));

        use std::io::Cursor;
        let mut dmx = open_wav_demuxer(Box::new(Cursor::new(bytes))).unwrap();
        assert_eq!(dmx.chna(), Some(&chna));
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:chna.0.uid"), Some(&"ATU_0000000A".to_string()));
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, payload);

        // No `chna` chunk → typed accessor is None.
        let plain = wav_with_smpl_and_inst(None, None);
        let dmx = open_wav_demuxer(Box::new(Cursor::new(plain))).unwrap();
        assert_eq!(dmx.chna(), None);
    }

    /// `AudioId` ADM reference classification (§1.2 prefix → kind) and the
    /// §3 common-vs-custom definition rule (trailing four hex digits
    /// `≤ 0x0FFF` = common BS.2094 def, `≥ 0x1000` = custom).
    #[test]
    fn chna_adm_ref_classification() {
        // AT_ trackFormat, common pack (trailing 0x0002 ≤ 0x0FFF).
        let common = audio_id(1, "ATU_00000001", "AT_00010001_01", "AP_00010002");
        assert_eq!(common.track_ref_kind(), AdmRefKind::TrackFormat);
        assert_eq!(common.pack_ref_kind(), AdmRefKind::PackFormat);
        // trackRef value low-16 = 0x0001 (≤ 0x0FFF) → common.
        assert_eq!(common.track_ref_scope(), Some(DefinitionScope::Common));
        // packRef value low-16 = 0x0002 (≤ 0x0FFF) → common.
        assert_eq!(common.pack_ref_scope(), Some(DefinitionScope::Common));

        // AC_ channelFormat ref (linear-PCM essence), custom pack
        // (trailing 0x1003 ≥ 0x1000), and a custom channel def
        // (0x1001 ≥ 0x1000).
        let custom = audio_id(2, "ATU_00000002", "AC_00011001_00", "AP_00011003");
        assert_eq!(custom.track_ref_kind(), AdmRefKind::ChannelFormat);
        assert_eq!(custom.track_ref_scope(), Some(DefinitionScope::Custom));
        assert_eq!(custom.pack_ref_scope(), Some(DefinitionScope::Custom));

        // Boundary: exactly 0x0FFF is still common; 0x1000 flips to custom.
        let edge_lo = audio_id(3, "ATU_00000003", "AT_00000FFF_01", "");
        assert_eq!(edge_lo.track_ref_scope(), Some(DefinitionScope::Common));
        let edge_hi = audio_id(4, "ATU_00000004", "AT_00001000_01", "");
        assert_eq!(edge_hi.track_ref_scope(), Some(DefinitionScope::Custom));

        // All-NUL pack (no pack required) → Unknown kind, no scope.
        assert_eq!(edge_lo.pack_ref_kind(), AdmRefKind::Unknown);
        assert_eq!(edge_lo.pack_ref_scope(), None);
    }

    /// The §3 classification reaches the surfaced `wav:chna.*` metadata:
    /// per defined record we expose `.track_ref_kind` / `.pack_ref_kind`
    /// and `.track_ref_definition` / `.pack_ref_definition`. Record 0 is a
    /// common BS.2094 channel def with no pack; record 1 is a custom
    /// trackFormat with a custom pack.
    #[test]
    fn chna_definition_scope_metadata() {
        let chna = ChnaChunk {
            num_tracks: 2,
            num_uids: 2,
            ids: vec![
                // Linear-PCM essence: AC_ channelFormat, common (0x0001),
                // no pack.
                audio_id(1, "ATU_00000001", "AC_00010001_00", ""),
                // Custom trackFormat (0x1001) + custom pack (0x1002).
                audio_id(2, "ATU_00000002", "AT_00011001_01", "AP_00011002"),
            ],
        };
        let bytes = wav_with_chna(&chna.to_bytes());
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        // Record 0: AC_ channelFormat, common, no pack keys.
        assert_eq!(
            md.get("wav:chna.0.track_ref_kind"),
            Some(&"audioChannelFormat".to_string())
        );
        assert_eq!(
            md.get("wav:chna.0.track_ref_definition"),
            Some(&"common".to_string())
        );
        assert_eq!(md.get("wav:chna.0.pack_ref"), None);
        assert_eq!(md.get("wav:chna.0.pack_ref_kind"), None);
        assert_eq!(md.get("wav:chna.0.pack_ref_definition"), None);

        // Record 1: AT_ trackFormat custom, AP_ pack custom.
        assert_eq!(
            md.get("wav:chna.1.track_ref_kind"),
            Some(&"audioTrackFormatID".to_string())
        );
        assert_eq!(
            md.get("wav:chna.1.track_ref_definition"),
            Some(&"custom".to_string())
        );
        assert_eq!(
            md.get("wav:chna.1.pack_ref_kind"),
            Some(&"audioPackFormatID".to_string())
        );
        assert_eq!(
            md.get("wav:chna.1.pack_ref_definition"),
            Some(&"custom".to_string())
        );
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `plst`
    /// chunk inserted between `fmt ` and `data`. Mirrors
    /// `wav_with_cue_and_adtl` but for the playlist chunk alone — the
    /// playlist references cue ids but is parsed independently of any
    /// preceding `cue ` chunk.
    fn wav_with_plst(plst_body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // plst chunk
        buf.extend_from_slice(b"plst");
        buf.extend_from_slice(&(plst_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(plst_body);
        if plst_body.len() % 2 == 1 {
            buf.push(0);
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Build a single 12-byte `<play-segment>` record per
    /// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3.
    fn plst_segment(dw_name: u32, dw_length: u32, dw_loops: u32) -> Vec<u8> {
        let mut b = Vec::with_capacity(12);
        b.extend_from_slice(&dw_name.to_le_bytes());
        b.extend_from_slice(&dw_length.to_le_bytes());
        b.extend_from_slice(&dw_loops.to_le_bytes());
        b
    }

    /// Full `plst` round-trip: three play segments referencing cue ids
    /// 1, 2, 1 (replaying cue 1) surface under index-keyed metadata.
    /// The replay case is the reason segments are indexed by position
    /// rather than by `dwName`.
    #[test]
    fn plst_full_metadata() {
        let mut plst_body = Vec::new();
        plst_body.extend_from_slice(&3u32.to_le_bytes()); // dwSegments
        plst_body.extend(plst_segment(1, 4410, 1)); // 0.1s of cue 1
        plst_body.extend(plst_segment(2, 8820, 2)); // 0.2s of cue 2, twice
        plst_body.extend(plst_segment(1, 4410, 1)); // replay cue 1

        let bytes = wav_with_plst(&plst_body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(md.get("wav:plst.count"), Some(&"3".to_string()));
        assert_eq!(md.get("wav:plst.0.cue_id"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:plst.0.length"), Some(&"4410".to_string()));
        assert_eq!(md.get("wav:plst.0.loops"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:plst.1.cue_id"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:plst.1.length"), Some(&"8820".to_string()));
        assert_eq!(md.get("wav:plst.1.loops"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:plst.2.cue_id"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:plst.2.length"), Some(&"4410".to_string()));
        assert_eq!(md.get("wav:plst.2.loops"), Some(&"1".to_string()));
    }

    /// A `plst` chunk whose `dwSegments` count exceeds the body length
    /// must not panic — the parser surfaces only the records that
    /// actually fit in the body (defensive against writers that lie
    /// about the count, matching the `cue ` clamp behaviour).
    #[test]
    fn plst_truncated_count_is_clamped() {
        // Claim 10 segments, ship 1.
        let mut plst_body = Vec::new();
        plst_body.extend_from_slice(&10u32.to_le_bytes());
        plst_body.extend(plst_segment(42, 1000, 1));
        let bytes = wav_with_plst(&plst_body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:plst.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:plst.0.cue_id"), Some(&"42".to_string()));
        // Stream still opens cleanly.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `plst` chunk shorter than the 4-byte `dwSegments` header is
    /// treated as opaque and skipped — no metadata keys emitted, stream
    /// still opens.
    #[test]
    fn plst_truncated_header_is_opaque() {
        let plst_body = vec![0u8, 0]; // < 4 bytes
        let bytes = wav_with_plst(&plst_body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(md.keys().all(|k| !k.starts_with("wav:plst.")));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A zero-segment `plst` chunk surfaces `wav:plst.count = 0` with
    /// no per-segment keys.
    #[test]
    fn plst_zero_segments() {
        let plst_body = 0u32.to_le_bytes().to_vec();
        let bytes = wav_with_plst(&plst_body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:plst.count"), Some(&"0".to_string()));
        assert!(md
            .keys()
            .all(|k| !k.starts_with("wav:plst.") || k == "wav:plst.count"));
    }

    /// An odd-length `plst` body forces a pad byte; the `data` chunk
    /// that follows must still be located correctly.
    #[test]
    fn plst_odd_body_padding() {
        // One 12-byte segment + an extra trailing byte → 17 bytes (odd).
        let mut plst_body = Vec::new();
        plst_body.extend_from_slice(&1u32.to_le_bytes());
        plst_body.extend(plst_segment(5, 100, 1));
        plst_body.push(0xAA);
        let bytes = wav_with_plst(&plst_body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:plst.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:plst.0.cue_id"), Some(&"5".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `fact`
    /// chunk inserted between `fmt ` and `data`. Mirrors
    /// `wav_with_plst` but for the `fact` chunk — exercised separately
    /// so the fact-chunk tests don't depend on the muxer also writing
    /// one (which the muxer skips for PCM by design).
    fn wav_with_fact(fact_body: &[u8], data_payload: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // fact chunk
        buf.extend_from_slice(b"fact");
        buf.extend_from_slice(&(fact_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(fact_body);
        if fact_body.len() % 2 == 1 {
            buf.push(0);
        }
        // data chunk with caller-supplied payload (S16 = 2 bytes/sample).
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&(data_payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(data_payload);
        if data_payload.len() % 2 == 1 {
            buf.push(0);
        }
        buf
    }

    /// Spec-minimum `fact` body — the 4-byte `dwFileSize` field. A PCM
    /// file with a `fact` chunk whose sample count matches the
    /// `data / block_align` heuristic should surface
    /// `wav:fact.sample_count` and *no* `wav:fact.mismatch` key. This
    /// is the well-formed case some WAV writers emit even for PCM
    /// (common for large files past the 2 GiB envelope, and emitted
    /// unconditionally by several DAW writers).
    #[test]
    fn fact_minimum_body_matches_data() {
        // 100 mono S16 samples → 200 data bytes; fact says 100 too.
        let payload: Vec<u8> = (0..200u32).map(|i| (i & 0xFF) as u8).collect();
        let fact_body = 100u32.to_le_bytes().to_vec();
        let bytes = wav_with_fact(&fact_body, &payload);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:fact.sample_count"), Some(&"100".to_string()));
        assert!(!md.contains_key("wav:fact.mismatch"));
        assert!(!md.contains_key("wav:fact.body_len"));
        // Duration reflects the fact-chunk sample count (here matching
        // the heuristic, so the public Duration:sample_count is 100).
        assert_eq!(dmx.streams()[0].duration, Some(100));
    }

    /// `fact` `dwFileSize` that disagrees with the `data / block_align`
    /// heuristic surfaces `wav:fact.mismatch` and the duration follows
    /// the fact value. This is the canonical compressed-WAV path
    /// (e.g. a hypothetical ADPCM stream whose nibble-packed `data`
    /// chunk yields fewer-than-bytes samples).
    #[test]
    fn fact_mismatch_surfaces_diagnostic_and_overrides_duration() {
        // 200 data bytes, fact claims only 50 samples → mismatch.
        let payload: Vec<u8> = (0..200u32).map(|i| (i & 0xFF) as u8).collect();
        let fact_body = 50u32.to_le_bytes().to_vec();
        let bytes = wav_with_fact(&fact_body, &payload);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:fact.sample_count"), Some(&"50".to_string()));
        assert_eq!(
            md.get("wav:fact.mismatch"),
            Some(&"block_samples=100 fact_samples=50".to_string())
        );
        assert_eq!(dmx.streams()[0].duration, Some(50));
    }

    /// A `fact` chunk longer than the spec-minimum 4 bytes is
    /// tolerated per RIFF MCI §3 ("Added fields will appear following
    /// the `dwFileSize` field. Applications can use the chunk size
    /// field to determine which fields are present.") — the parser
    /// reads `dwFileSize`, surfaces `wav:fact.body_len` so callers
    /// can see extension bytes are present, and ignores the rest. The
    /// `data` chunk that follows must still be located correctly.
    #[test]
    fn fact_extension_bytes_preserved_in_body_len() {
        // 200 data bytes; fact has 4 + 8 = 12 bytes (future-extension
        // bytes are opaque per spec but the body_len surfaces them).
        let payload: Vec<u8> = (0..200u32).map(|i| (i & 0xFF) as u8).collect();
        let mut fact_body = 100u32.to_le_bytes().to_vec();
        fact_body.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04]);
        let bytes = wav_with_fact(&fact_body, &payload);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:fact.sample_count"), Some(&"100".to_string()));
        assert_eq!(md.get("wav:fact.body_len"), Some(&"12".to_string()));
        // Stream still opens cleanly — the 12-byte fact body is even
        // so no pad byte; the data chunk that follows is intact.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `fact` chunk shorter than the 4-byte fixed `dwFileSize` field
    /// is treated as opaque and skipped — no metadata keys emitted,
    /// the stream still opens cleanly. The `data` chunk that follows
    /// must still be located (validates the chunk-walk pad-byte
    /// arithmetic for the < 4-byte case).
    #[test]
    fn fact_truncated_body_is_opaque() {
        let payload: Vec<u8> = vec![0u8, 0u8];
        let fact_body = vec![0u8, 0u8]; // < 4 bytes → opaque
        let bytes = wav_with_fact(&fact_body, &payload);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(md.keys().all(|k| !k.starts_with("wav:fact.")));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// An odd-length `fact` body forces a pad byte; the `data` chunk
    /// that follows must still be located correctly (regression guard
    /// matching `plst_odd_body_padding`). RIFF MCI §2 "Chunks"
    /// requires all chunks to be word-aligned, with an implicit pad
    /// byte appended when the body is odd. The pad byte is NOT part
    /// of the body length carried in the chunk header.
    #[test]
    fn fact_odd_body_padding() {
        // 4-byte dwFileSize + 1 future-extension byte = 5 bytes (odd).
        let payload: Vec<u8> = vec![0xAA, 0x55];
        let mut fact_body = 1u32.to_le_bytes().to_vec();
        fact_body.push(0x42);
        let bytes = wav_with_fact(&fact_body, &payload);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:fact.sample_count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:fact.body_len"), Some(&"5".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Round-trip via the muxer: an A-law stream must carry a `fact`
    /// chunk per RIFF MCI §3 ("The 'fact' chunk is required ... for
    /// all compressed audio formats"). The demuxer surfaces the
    /// sample count under `wav:fact.sample_count` and the duration
    /// reflects it. For G.711 mono one byte == one per-channel
    /// sample so the value matches the heuristic — the mismatch key
    /// must be absent.
    #[test]
    fn fact_chunk_round_trip_alaw_mono() {
        let payload: Vec<u8> = (0..=255u8).collect(); // 256 mono A-law samples
        let stream = make_g711_stream("pcm_alaw", 1, 8_000);
        let bytes = mux_to_bytes(&stream, &payload, WavMuxOptions::default(), "alaw-fact");
        // The muxer should have emitted a `fact` chunk between fmt and data.
        assert!(
            find_chunk(&bytes, b"fact").is_some(),
            "muxer must emit fact chunk for non-PCM wFormatTag"
        );
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:fact.sample_count"), Some(&"256".to_string()));
        assert!(!md.contains_key("wav:fact.mismatch"));
        assert_eq!(dmx.streams()[0].duration, Some(256));
    }

    /// Round-trip via the muxer: a plain PCM stream MUST NOT carry a
    /// `fact` chunk (per RIFF MCI §3 "The chunk is not required for
    /// PCM files using the 'data' chunk format") — we skip emitting
    /// it to keep the post-r193 PCM muxer output byte-identical to
    /// pre-r193. Regression guard against accidentally emitting it
    /// for `wFormatTag = WAVE_FORMAT_PCM`.
    #[test]
    fn fact_chunk_not_emitted_for_pcm() {
        let samples: Vec<i16> = (0..100).map(|i| (i * 100) as i16).collect();
        let mut payload = Vec::with_capacity(samples.len() * 2);
        for s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }
        let stream = make_stream(SampleFormat::S16, 1, 8_000);
        let bytes = mux_to_bytes(&stream, &payload, WavMuxOptions::default(), "pcm-no-fact");
        assert!(
            find_chunk(&bytes, b"fact").is_none(),
            "PCM muxer output must not carry a fact chunk"
        );
    }

    /// Round-trip via the muxer: an EXTENSIBLE stream carries a `fact`
    /// chunk too — compliant readers dispatch on the on-wire
    /// `wFormatTag` first (which is `0xFFFE`, not PCM), so the chunk
    /// is required regardless of which SubFormat GUID the muxer
    /// selects.
    #[test]
    fn fact_chunk_round_trip_extensible() {
        let samples: Vec<i16> = (0..200).map(|i| (i * 50) as i16).collect();
        let mut payload = Vec::with_capacity(samples.len() * 2);
        for s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }
        let stream = make_stream(SampleFormat::S16, 1, 8_000);
        let opts = WavMuxOptions::default().with_extensible(0x4); // SPEAKER_FRONT_CENTER
        let bytes = mux_to_bytes(&stream, &payload, opts, "ext-fact");
        assert!(
            find_chunk(&bytes, b"fact").is_some(),
            "EXTENSIBLE muxer output must carry a fact chunk"
        );
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:fact.sample_count"), Some(&"200".to_string()));
        assert!(!md.contains_key("wav:fact.mismatch"));
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `iXML`
    /// chunk inserted between `fmt ` and `data`. Mirrors `wav_with_fact`
    /// for the third-party metadata block documented in
    /// `docs/container/riff/metadata/exiftool-riff-tags.html` § `iXML`.
    fn wav_with_ixml(ixml_body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // iXML chunk.
        buf.extend_from_slice(b"iXML");
        buf.extend_from_slice(&(ixml_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(ixml_body);
        if ixml_body.len() % 2 == 1 {
            buf.push(0);
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// A canonical Sound-Devices-style `iXML` document round-trips: the
    /// XML text surfaces verbatim under `wav:ixml`, and the raw chunk-
    /// body length surfaces under `wav:ixml.body_len`. The stream still
    /// opens cleanly.
    #[test]
    fn ixml_canonical_document_round_trips() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<BWFXML>
  <IXML_VERSION>2.10</IXML_VERSION>
  <PROJECT>OxideAV Round 205</PROJECT>
  <SCENE>scn-001</SCENE>
  <TAKE>1</TAKE>
  <NOTE>iXML canonical fixture</NOTE>
  <TRACK_LIST>
    <TRACK_COUNT>1</TRACK_COUNT>
    <TRACK>
      <CHANNEL_INDEX>1</CHANNEL_INDEX>
      <NAME>Boom</NAME>
      <FUNCTION>Dialog</FUNCTION>
    </TRACK>
  </TRACK_LIST>
</BWFXML>"#;
        let bytes = wav_with_ixml(xml.as_bytes());
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:ixml.body_len"),
            Some(&xml.len().to_string()),
            "raw chunk-body length must surface verbatim"
        );
        assert_eq!(
            md.get("wav:ixml").map(|s| s.as_str()),
            Some(xml.trim()),
            "iXML text payload must round-trip"
        );
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A NUL-padded `iXML` body (writers commonly reserve a fixed-size
    /// block then NUL-fill the trailing space) surfaces only the
    /// pre-NUL text under `wav:ixml`; the raw `body_len` still reflects
    /// the on-wire byte count so the trailing reserved bytes are not
    /// silently lost.
    #[test]
    fn ixml_trailing_nuls_trimmed_in_text_but_body_len_kept() {
        let mut body = b"<BWFXML><PROJECT>OAV</PROJECT></BWFXML>".to_vec();
        // Reserve another 64 bytes of NUL pad — emulates writers that
        // size the iXML region for in-place editing.
        body.resize(body.len() + 64, 0);
        let bytes = wav_with_ixml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:ixml"),
            Some(&"<BWFXML><PROJECT>OAV</PROJECT></BWFXML>".to_string())
        );
        assert_eq!(md.get("wav:ixml.body_len"), Some(&body.len().to_string()));
    }

    /// An `iXML` chunk whose body is empty (zero bytes between the
    /// 8-byte header and the next chunk) surfaces `wav:ixml.body_len = 0`
    /// but no `wav:ixml` text key. Defensive against writers that emit a
    /// placeholder iXML header without filling it.
    #[test]
    fn ixml_empty_body_surfaces_only_body_len() {
        let bytes = wav_with_ixml(&[]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:ixml.body_len"), Some(&"0".to_string()));
        assert!(!md.contains_key("wav:ixml"));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// An `iXML` chunk whose body is entirely NUL / whitespace (e.g.
    /// "padding awaiting a writer") surfaces `body_len` but no text key.
    #[test]
    fn ixml_whitespace_only_body_omits_text_key() {
        let body = b"   \t\r\n   \0\0\0".to_vec();
        let bytes = wav_with_ixml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:ixml.body_len"), Some(&body.len().to_string()));
        assert!(!md.contains_key("wav:ixml"));
    }

    /// An odd-length `iXML` body forces a 1-byte pad; the `data` chunk
    /// that follows must still be located correctly (regression guard
    /// matching `plst_odd_body_padding` / `fact_odd_body_padding`).
    #[test]
    fn ixml_odd_body_padding() {
        // 17 bytes of XML — odd, so the chunk-walk pad-byte path is
        // exercised.
        let body = b"<X>1</X><Y>2</Y>!".to_vec();
        assert_eq!(body.len() % 2, 1);
        let bytes = wav_with_ixml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:ixml"), Some(&"<X>1</X><Y>2</Y>!".to_string()));
        assert_eq!(md.get("wav:ixml.body_len"), Some(&"17".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `<axml>`
    /// chunk inserted between `fmt ` and `data`. Mirrors `wav_with_ixml`
    /// for the BWF supplement-5 XML metadata block documented in
    /// `docs/container/riff/metadata/ebu-tech3285s5-ADM.pdf` §3.
    fn wav_with_axml(axml_body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // axml chunk.
        buf.extend_from_slice(b"axml");
        buf.extend_from_slice(&(axml_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(axml_body);
        if axml_body.len() % 2 == 1 {
            buf.push(0);
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// A canonical EBUCore-wrapped ADM document round-trips: the XML
    /// text surfaces verbatim under `wav:axml`, and the raw chunk-body
    /// length surfaces under `wav:axml.body_len`. The fixture is the
    /// `<axml>` payload pattern from
    /// `docs/container/riff/metadata/ebu-tech3285s5-ADM.pdf` §4.2 with
    /// the inner element set trimmed to a single
    /// `<audioProgramme>` reference — enough to exercise the parser
    /// without dragging the full HOA pack into the fixture.
    #[test]
    fn axml_canonical_ebucore_adm_document_round_trips() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ebuCoreMain xmlns="urn:ebu:metadata-schema:ebucore"
    xmlns:dc="http://purl.org/dc/elements/1.1/">
  <coreMetadata>
    <format>
      <audioFormatExtended>
        <audioProgramme audioProgrammeID="APR_1001"
            audioProgrammeName="OxideAV Round 258 demo">
          <audioContentIDRef>ACO_1001</audioContentIDRef>
        </audioProgramme>
      </audioFormatExtended>
    </format>
  </coreMetadata>
</ebuCoreMain>"#;
        let bytes = wav_with_axml(xml.as_bytes());
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:axml.body_len"),
            Some(&xml.len().to_string()),
            "raw chunk-body length must surface verbatim"
        );
        assert_eq!(
            md.get("wav:axml").map(|s| s.as_str()),
            Some(xml.trim()),
            "axml text payload must round-trip"
        );
        // The chunk-walk must still resolve fmt + data after the
        // axml hop — regression guard matching ixml_canonical.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// An ISRC identifier `<axml>` payload (§4.1 example) round-trips
    /// just like the ADM one — the parser is schema-agnostic.
    #[test]
    fn axml_isrc_identifier_document_round_trips() {
        let xml = r#"<ebuCoreMain xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns="urn:ebu:metadata-schema:ebucore">
  <coreMetadata>
    <identifier typeLabel="GUID" typeDefinition="Globally Unique Identifier"
        formatLabel="ISRC" formatDefinition="International Standard Recording Code">
      <dc:identifier>ISRC:NOX001212345</dc:identifier>
    </identifier>
  </coreMetadata>
</ebuCoreMain>"#;
        let bytes = wav_with_axml(xml.as_bytes());
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:axml").map(|s| s.as_str()), Some(xml.trim()));
        assert!(
            md.get("wav:axml")
                .map(|s| s.contains("ISRC:NOX001212345"))
                .unwrap_or(false),
            "ISRC identifier must survive the round-trip"
        );
    }

    /// A NUL-padded `<axml>` body (writers commonly reserve a
    /// fixed-size block then NUL-fill the trailing space to keep the
    /// ADM document mutable in-place) surfaces only the pre-NUL text
    /// under `wav:axml`; the raw `body_len` still reflects the on-wire
    /// byte count so the trailing reserved bytes are not silently
    /// lost. Mirrors `ixml_trailing_nuls_trimmed_in_text_but_body_len_kept`.
    #[test]
    fn axml_trailing_nuls_trimmed_in_text_but_body_len_kept() {
        let mut body = b"<ebuCoreMain><id>X</id></ebuCoreMain>".to_vec();
        body.resize(body.len() + 128, 0);
        let bytes = wav_with_axml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:axml"),
            Some(&"<ebuCoreMain><id>X</id></ebuCoreMain>".to_string())
        );
        assert_eq!(md.get("wav:axml.body_len"), Some(&body.len().to_string()));
    }

    /// An `<axml>` chunk whose body is empty (zero bytes between the
    /// 8-byte header and the next chunk) surfaces
    /// `wav:axml.body_len = 0` but no `wav:axml` text key. Defensive
    /// against writers that emit a placeholder header without filling
    /// it; the §3 "shall be ignored" rule for unintelligible content
    /// is a schema-level concern, not a byte-level one — the body
    /// length stays observable so the placeholder is discoverable.
    #[test]
    fn axml_empty_body_surfaces_only_body_len() {
        let bytes = wav_with_axml(&[]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:axml.body_len"), Some(&"0".to_string()));
        assert!(!md.contains_key("wav:axml"));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// An `<axml>` chunk whose body is entirely NUL / whitespace
    /// (placeholder reserved by a writer ahead of an ADM authoring
    /// pass) surfaces `body_len` but no text key.
    #[test]
    fn axml_whitespace_only_body_omits_text_key() {
        let body = b"   \t\r\n   \0\0\0".to_vec();
        let bytes = wav_with_axml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:axml.body_len"), Some(&body.len().to_string()));
        assert!(!md.contains_key("wav:axml"));
    }

    /// An odd-length `<axml>` body forces a 1-byte pad; the `data`
    /// chunk that follows must still be located correctly (regression
    /// guard matching `ixml_odd_body_padding`).
    #[test]
    fn axml_odd_body_padding() {
        // 25 bytes of XML — odd, so the chunk-walk pad-byte path is
        // exercised. The inner content is deliberately short but
        // schema-recognisable.
        let body = b"<root><id>z</id></root>!!".to_vec();
        assert_eq!(body.len() % 2, 1);
        let bytes = wav_with_axml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:axml"),
            Some(&"<root><id>z</id></root>!!".to_string())
        );
        assert_eq!(md.get("wav:axml.body_len"), Some(&"25".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `<bxml>`
    /// chunk inserted between `fmt ` and `data`. The body is the on-wire
    /// `bxml` payload (2-byte `fmtType` header + XML data), per ITU-R
    /// BS.2088-2 §6 (`docs/container/riff/metadata/R-REC-BS.2088.pdf`).
    fn wav_with_bxml(bxml_body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // bxml chunk.
        buf.extend_from_slice(b"bxml");
        buf.extend_from_slice(&(bxml_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(bxml_body);
        if bxml_body.len() % 2 == 1 {
            buf.push(0);
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Helper: build a `<bxml>` on-wire body = LE `fmtType` WORD followed
    /// by the raw `payload` bytes (the XML data, compressed or not).
    fn bxml_body(fmt_type: u16, payload: &[u8]) -> Vec<u8> {
        let mut b = fmt_type.to_le_bytes().to_vec();
        b.extend_from_slice(payload);
        b
    }

    /// An uncompressed (`fmtType == 0x0000`) `<bxml>` chunk surfaces its
    /// XML text under `wav:bxml`, the `none` compression label, the raw
    /// `0x0000` `fmt_type`, and a `body_len` that includes the 2-byte
    /// header — per ITU-R BS.2088-2 §6.2.
    #[test]
    fn bxml_uncompressed_surfaces_xml_text() {
        let xml = r#"<ebuCoreMain xmlns="urn:ebu:metadata-schema:ebucore">
  <coreMetadata>
    <format>
      <audioFormatExtended>
        <audioProgramme audioProgrammeID="APR_1001"/>
      </audioFormatExtended>
    </format>
  </coreMetadata>
</ebuCoreMain>"#;
        let body = bxml_body(0x0000, xml.as_bytes());
        let bytes = wav_with_bxml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:bxml.body_len"), Some(&body.len().to_string()));
        assert_eq!(md.get("wav:bxml.fmt_type"), Some(&"0x0000".to_string()));
        assert_eq!(md.get("wav:bxml.compression"), Some(&"none".to_string()));
        assert_eq!(md.get("wav:bxml").map(|s| s.as_str()), Some(xml.trim()));
        // chunk-walk still resolves fmt + data after the bxml hop.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A gzip-flagged (`fmtType == 0x0001`) `<bxml>` chunk surfaces the
    /// `gzip` compression label and raw `fmt_type` but does NOT attempt
    /// to expose `wav:bxml` text — the container layer leaves RFC 1952
    /// inflation to a higher-level ADM-aware consumer (§6.2). The
    /// payload bytes here are an opaque (non-XML) gzip stub: the parser
    /// must not choke on them.
    #[test]
    fn bxml_gzip_surfaces_header_but_not_text() {
        // gzip magic 1f 8b + deflate method 08 + flags/mtime stub.
        let body = bxml_body(0x0001, &[0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00]);
        let bytes = wav_with_bxml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:bxml.body_len"), Some(&body.len().to_string()));
        assert_eq!(md.get("wav:bxml.fmt_type"), Some(&"0x0001".to_string()));
        assert_eq!(md.get("wav:bxml.compression"), Some(&"gzip".to_string()));
        assert!(
            !md.contains_key("wav:bxml"),
            "compressed payload must not surface as text"
        );
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `<bxml>` chunk with an undocumented (private/future) `fmtType`
    /// surfaces the raw `fmt_type` but omits the `compression` label
    /// (the raw value is the unambiguous source of truth) and surfaces
    /// no text.
    #[test]
    fn bxml_unknown_fmt_type_omits_compression_label() {
        let body = bxml_body(0x00FF, b"\x01\x02\x03\x04");
        let bytes = wav_with_bxml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:bxml.fmt_type"), Some(&"0x00FF".to_string()));
        assert!(!md.contains_key("wav:bxml.compression"));
        assert!(!md.contains_key("wav:bxml"));
    }

    /// A NUL-padded uncompressed `<bxml>` body (a writer reserving a
    /// fixed-size block for later in-place ADM editing) surfaces only the
    /// pre-NUL text; `body_len` still reflects the full on-wire span so
    /// the reserved region is observable. Mirrors the `<axml>` contract.
    #[test]
    fn bxml_uncompressed_trailing_nuls_trimmed_body_len_kept() {
        let mut payload = b"<root><id>X</id></root>".to_vec();
        payload.resize(payload.len() + 64, 0);
        let body = bxml_body(0x0000, &payload);
        let bytes = wav_with_bxml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:bxml"),
            Some(&"<root><id>X</id></root>".to_string())
        );
        assert_eq!(md.get("wav:bxml.body_len"), Some(&body.len().to_string()));
    }

    /// A `<bxml>` chunk whose body is shorter than the 2-byte `fmtType`
    /// header is skipped-as-opaque: only `wav:bxml.body_len` surfaces,
    /// and the chunk-walk still finds the following `data` chunk.
    #[test]
    fn bxml_truncated_header_skipped_as_opaque() {
        // 1-byte body (odd → exercises the pad-byte path too).
        let body = vec![0x00u8];
        let bytes = wav_with_bxml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:bxml.body_len"), Some(&"1".to_string()));
        assert!(!md.contains_key("wav:bxml.fmt_type"));
        assert!(!md.contains_key("wav:bxml.compression"));
        assert!(!md.contains_key("wav:bxml"));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// An uncompressed `<bxml>` whose XML payload is empty (header only)
    /// surfaces `body_len = 2`, the `fmt_type` / `compression` header
    /// keys, but no `wav:bxml` text key.
    #[test]
    fn bxml_uncompressed_empty_payload_omits_text() {
        let body = bxml_body(0x0000, &[]);
        let bytes = wav_with_bxml(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:bxml.body_len"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:bxml.fmt_type"), Some(&"0x0000".to_string()));
        assert_eq!(md.get("wav:bxml.compression"), Some(&"none".to_string()));
        assert!(!md.contains_key("wav:bxml"));
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `_PMX`
    /// (XMP packet) chunk inserted between `fmt ` and `data`. Mirrors
    /// `wav_with_axml` / `wav_with_ixml` for the third-party XMP
    /// metadata block catalogued in
    /// `docs/container/riff/metadata/exiftool-riff-tags.html` § "RIFF
    /// Main tags" (entry `'_PMX'`, scope "AVI and WAV files").
    fn wav_with_pmx(pmx_body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // _PMX chunk.
        buf.extend_from_slice(b"_PMX");
        buf.extend_from_slice(&(pmx_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(pmx_body);
        if pmx_body.len() % 2 == 1 {
            buf.push(0);
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// A canonical Adobe-style XMP packet round-trips: the UTF-8 XMP
    /// text surfaces verbatim under `wav:xmp`, and the raw chunk-body
    /// length surfaces under `wav:xmp.body_len`. The wrapping
    /// `<?xpacket begin=...?>` / `<?xpacket end=...?>` processing
    /// instructions are passed through unchanged — the parser is
    /// schema-agnostic by design.
    #[test]
    fn pmx_canonical_xmp_packet_round_trips() {
        let xml = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">OxideAV Round 263 Fixture</rdf:li>
        </rdf:Alt>
      </dc:title>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;
        let bytes = wav_with_pmx(xml.as_bytes());
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:xmp.body_len"),
            Some(&xml.len().to_string()),
            "raw chunk-body length must surface verbatim"
        );
        assert_eq!(
            md.get("wav:xmp").map(|s| s.as_str()),
            Some(xml.trim()),
            "XMP packet text must round-trip"
        );
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A NUL-padded `_PMX` body (writers commonly reserve a fixed-size
    /// block for in-place XMP editing) surfaces only the pre-NUL text
    /// under `wav:xmp`; the raw `body_len` still reflects the on-wire
    /// byte count so the trailing reserved bytes are observable.
    #[test]
    fn pmx_trailing_nuls_trimmed_in_text_but_body_len_kept() {
        let mut body = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>".to_vec();
        // Reserve another 96 bytes of NUL pad — emulates writers that
        // size the XMP region for in-place editing.
        body.resize(body.len() + 96, 0);
        let bytes = wav_with_pmx(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:xmp"),
            Some(&"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>".to_string())
        );
        assert_eq!(md.get("wav:xmp.body_len"), Some(&body.len().to_string()));
    }

    /// A `_PMX` chunk whose body is empty (zero bytes between the
    /// 8-byte header and the next chunk) surfaces `wav:xmp.body_len = 0`
    /// but no `wav:xmp` text key. Defensive against writers that emit a
    /// placeholder XMP header without filling it.
    #[test]
    fn pmx_empty_body_surfaces_only_body_len() {
        let bytes = wav_with_pmx(&[]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:xmp.body_len"), Some(&"0".to_string()));
        assert!(!md.contains_key("wav:xmp"));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `_PMX` chunk whose body is entirely NUL / whitespace
    /// (placeholder awaiting an XMP-aware writer) surfaces `body_len`
    /// but no text key.
    #[test]
    fn pmx_whitespace_only_body_omits_text_key() {
        let body = b"   \t\r\n   \0\0\0".to_vec();
        let bytes = wav_with_pmx(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:xmp.body_len"), Some(&body.len().to_string()));
        assert!(!md.contains_key("wav:xmp"));
    }

    /// An odd-length `_PMX` body forces a 1-byte pad; the `data` chunk
    /// that follows must still be located correctly (regression guard
    /// matching `axml_odd_body_padding` / `ixml_odd_body_padding`).
    #[test]
    fn pmx_odd_body_padding() {
        // 27 bytes — odd, exercises the chunk-walk pad-byte path.
        let body = b"<x:xmpmeta xmlns:x=\"a:n\"/>!".to_vec();
        assert_eq!(body.len() % 2, 1);
        let bytes = wav_with_pmx(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(
            md.get("wav:xmp"),
            Some(&"<x:xmpmeta xmlns:x=\"a:n\"/>!".to_string())
        );
        assert_eq!(md.get("wav:xmp.body_len"), Some(&"27".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A file with no `_PMX` chunk surfaces no `wav:xmp.*` keys at all
    /// — absence is observable. Mirrors the matching absence guards for
    /// `iXML`, `axml`, and `JUNK`.
    #[test]
    fn pmx_absent_chunk_emits_no_xmp_keys() {
        // Reuse the wav_with_axml helper but pass through with NO axml
        // body to avoid coincidental key emissions — we want a file with
        // neither axml nor _PMX, which the iXML/JUNK suites already do.
        // Build the minimal "fmt + data only" PCM file directly.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        let dmx = open_demux_from_bytes(buf);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(!md.contains_key("wav:xmp"));
        assert!(!md.contains_key("wav:xmp.body_len"));
    }

    /// Build a minimal valid PCM WAV with a caller-supplied raw `CSET`
    /// chunk inserted between `fmt ` and `data`. Mirrors `wav_with_ixml`
    /// for the character-set declaration documented in
    /// `docs/container/riff/metadata/microsoft-riffmci.pdf` §3
    /// "CSET (Character Set) Chunk".
    fn wav_with_cset(cset_body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // CSET chunk.
        buf.extend_from_slice(b"CSET");
        buf.extend_from_slice(&(cset_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(cset_body);
        if cset_body.len() % 2 == 1 {
            buf.push(0);
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Build a canonical 8-byte CSET body from its four `u16` fields,
    /// mirroring the spec layout `(code_page, country, language, dialect)`.
    fn cset_body(code_page: u16, country: u16, language: u16, dialect: u16) -> Vec<u8> {
        let mut body = Vec::with_capacity(8);
        body.extend_from_slice(&code_page.to_le_bytes());
        body.extend_from_slice(&country.to_le_bytes());
        body.extend_from_slice(&language.to_le_bytes());
        body.extend_from_slice(&dialect.to_le_bytes());
        body
    }

    /// A canonical CSET chunk for Windows-1252 / UK English / United
    /// Kingdom round-trips: every raw field surfaces under the matching
    /// `wav:cset.*` key, the human-readable lookups resolve, and the
    /// `body_len` reflects the 8-byte canonical struct.
    #[test]
    fn cset_canonical_uk_english_round_trips() {
        // wCodePage = 1252 (Windows Western European), wCountryCode = 44
        // (United Kingdom), wLanguageCode = 9 / wDialect = 2 (UK English).
        let body = cset_body(1252, 44, 9, 2);
        let bytes = wav_with_cset(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cset.body_len"), Some(&"8".to_string()));
        assert_eq!(md.get("wav:cset.code_page"), Some(&"1252".to_string()));
        assert_eq!(md.get("wav:cset.country"), Some(&"44".to_string()));
        assert_eq!(
            md.get("wav:cset.country_name"),
            Some(&"United Kingdom".to_string())
        );
        assert_eq!(md.get("wav:cset.language"), Some(&"9".to_string()));
        assert_eq!(md.get("wav:cset.dialect"), Some(&"2".to_string()));
        assert_eq!(
            md.get("wav:cset.language_name"),
            Some(&"UK English".to_string())
        );
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// All-zero CSET — the spec-mandated "use defaults" form — must
    /// surface the raw zeros plus the human-readable "None" placeholders
    /// from the country / language tables. The language pair `(0, _)`
    /// resolves to `None` per the §3 enumeration.
    #[test]
    fn cset_all_zero_uses_spec_defaults() {
        let body = cset_body(0, 0, 0, 0);
        let bytes = wav_with_cset(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cset.code_page"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:cset.country"), Some(&"0".to_string()));
        assert_eq!(
            md.get("wav:cset.country_name"),
            Some(&"None".to_string()),
            "wCountryCode = 0 must resolve to the §3 'None' placeholder"
        );
        assert_eq!(md.get("wav:cset.language"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:cset.dialect"), Some(&"0".to_string()));
        assert_eq!(
            md.get("wav:cset.language_name"),
            Some(&"None".to_string()),
            "wLanguageCode = 0 must resolve to the §3 'None' placeholder regardless of dialect"
        );
    }

    /// Out-of-table code-page / country / language values still surface
    /// their raw numeric value; the human-readable lookups are simply
    /// absent. Defensive guard for vendor extensions and future code
    /// pages (e.g. 65001 / UTF-8 is not in the 1991 enumeration).
    #[test]
    fn cset_unknown_codes_emit_raw_values_only() {
        // 65001 (UTF-8 / not in the §3 enumeration), 999 (not a defined
        // country code), 99 / 99 (not a defined language pair).
        let body = cset_body(65001, 999, 99, 99);
        let bytes = wav_with_cset(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cset.code_page"), Some(&"65001".to_string()));
        assert_eq!(md.get("wav:cset.country"), Some(&"999".to_string()));
        assert!(
            !md.contains_key("wav:cset.country_name"),
            "unknown country must not synthesise a human-readable name"
        );
        assert_eq!(md.get("wav:cset.language"), Some(&"99".to_string()));
        assert_eq!(md.get("wav:cset.dialect"), Some(&"99".to_string()));
        assert!(
            !md.contains_key("wav:cset.language_name"),
            "unknown language pair must not synthesise a human-readable name"
        );
    }

    /// A CSET body shorter than the canonical 8-byte struct is treated
    /// as opaque: only `wav:cset.body_len` is emitted. Defensive against
    /// truncated writers; the chunk-walk loop still advances correctly.
    #[test]
    fn cset_short_body_treated_as_opaque() {
        // 4 bytes — half the spec struct. No `code_page` / `country` /
        // language pair should surface (incomplete fields would be a
        // guess, not a read).
        let body = vec![0x52, 0x04, 0x00, 0x00];
        let bytes = wav_with_cset(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cset.body_len"), Some(&"4".to_string()));
        assert!(!md.contains_key("wav:cset.code_page"));
        assert!(!md.contains_key("wav:cset.country"));
        assert!(!md.contains_key("wav:cset.language"));
        assert!(!md.contains_key("wav:cset.dialect"));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A CSET body longer than the canonical 8 bytes tolerates the
    /// trailing region (forward-compat) — every documented field still
    /// surfaces, and `body_len` reflects the actual on-wire size so the
    /// extra payload is observable.
    #[test]
    fn cset_oversized_body_tolerates_trailing_bytes() {
        // 8-byte canonical struct + 4 trailing bytes a hypothetical
        // future extension might reserve.
        let mut body = cset_body(932, 81, 17, 1); // Shift-JIS / Japan / Japanese
        body.extend_from_slice(&[0xFE, 0xCA, 0xAD, 0xDE]);
        assert_eq!(body.len(), 12);
        let bytes = wav_with_cset(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cset.body_len"), Some(&"12".to_string()));
        assert_eq!(md.get("wav:cset.code_page"), Some(&"932".to_string()));
        assert_eq!(md.get("wav:cset.country"), Some(&"81".to_string()));
        assert_eq!(md.get("wav:cset.country_name"), Some(&"Japan".to_string()));
        assert_eq!(md.get("wav:cset.language"), Some(&"17".to_string()));
        assert_eq!(md.get("wav:cset.dialect"), Some(&"1".to_string()));
        assert_eq!(
            md.get("wav:cset.language_name"),
            Some(&"Japanese".to_string())
        );
    }

    /// CSET coexists with `LIST INFO` without disrupting the existing
    /// INFO sub-ID parser — the CSET fields surface alongside the INFO
    /// title and the file still parses end-to-end. Regression guard for
    /// the chunk-walk ordering CSET → LIST(INFO).
    #[test]
    fn cset_coexists_with_list_info() {
        // Hand-build: RIFF / WAVE / fmt / CSET / LIST(INFO INAM "T") /
        // data(empty). CSET says Windows-1252 / France / French.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // CSET: 1252 / 33 (France) / 12 / 1 (French).
        let cset = cset_body(1252, 33, 12, 1);
        buf.extend_from_slice(b"CSET");
        buf.extend_from_slice(&(cset.len() as u32).to_le_bytes());
        buf.extend_from_slice(&cset);
        // LIST INFO with INAM = "T" (1 byte, odd-length → 1 byte pad).
        // INFO header (4) + INAM(4) + size(4) + payload(1) + pad(1) = 14.
        // Total LIST body = "INFO" + (INAM + size + "T" + pad) = 4 + 10 = 14.
        let mut list_body = Vec::new();
        list_body.extend_from_slice(b"INFO");
        list_body.extend_from_slice(b"INAM");
        list_body.extend_from_slice(&1u32.to_le_bytes());
        list_body.extend_from_slice(b"T");
        list_body.push(0); // ZSTR NUL terminator (consumed as the in-body pad).
        buf.extend_from_slice(b"LIST");
        buf.extend_from_slice(&(list_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(&list_body);
        // empty data chunk.
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());

        let dmx = open_demux_from_bytes(buf);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cset.code_page"), Some(&"1252".to_string()));
        assert_eq!(md.get("wav:cset.country_name"), Some(&"France".to_string()));
        assert_eq!(
            md.get("wav:cset.language_name"),
            Some(&"French".to_string())
        );
        assert_eq!(md.get("title"), Some(&"T".to_string()));
    }

    /// An odd-length CSET body forces a 1-byte pad; the `data` chunk
    /// that follows must still be located correctly (regression guard
    /// matching `ixml_odd_body_padding`).
    #[test]
    fn cset_odd_body_padding() {
        // 9-byte CSET body: 8 canonical bytes + 1 trailing sentinel.
        let mut body = cset_body(1252, 1, 9, 1);
        body.push(0xAA);
        assert_eq!(body.len() % 2, 1);
        let bytes = wav_with_cset(&body);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:cset.body_len"), Some(&"9".to_string()));
        assert_eq!(md.get("wav:cset.code_page"), Some(&"1252".to_string()));
        assert_eq!(md.get("wav:cset.country_name"), Some(&"USA".to_string()));
        assert_eq!(
            md.get("wav:cset.language_name"),
            Some(&"US English".to_string())
        );
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// `cset_country_name` covers the spec's enumerated country codes;
    /// unknown codes return `None`. Spot-check the boundary codes (the
    /// three-digit Portugal / Luxembourg / Iceland / Finland entries)
    /// plus a representative one-digit code (USA).
    #[test]
    fn cset_country_name_table_spot_checks() {
        assert_eq!(cset_country_name(1), Some("USA"));
        assert_eq!(cset_country_name(44), Some("United Kingdom"));
        assert_eq!(cset_country_name(351), Some("Portugal"));
        assert_eq!(cset_country_name(358), Some("Finland"));
        assert_eq!(cset_country_name(0), Some("None"));
        assert_eq!(cset_country_name(500), None);
    }

    /// `cset_language_name` covers the spec's enumerated `(language,
    /// dialect)` pairs. Spot-check the dialect-disambiguated rows
    /// (English UK/US, French Belgian/Canadian/Swiss, Serbo-Croatian
    /// Latin/Cyrillic) — they are the entries the table exists *for*.
    #[test]
    fn cset_language_name_table_spot_checks() {
        assert_eq!(cset_language_name(9, 1), Some("US English"));
        assert_eq!(cset_language_name(9, 2), Some("UK English"));
        assert_eq!(cset_language_name(12, 2), Some("Belgian French"));
        assert_eq!(cset_language_name(12, 3), Some("Canadian French"));
        assert_eq!(cset_language_name(12, 4), Some("Swiss French"));
        assert_eq!(cset_language_name(26, 1), Some("Serbo-Croatian (Latin)"));
        assert_eq!(cset_language_name(26, 2), Some("Serbo-Croatian (Cyrillic)"));
        assert_eq!(cset_language_name(0, 0), Some("None"));
        assert_eq!(cset_language_name(0, 1), Some("None"));
        // Defined language, undefined dialect — must NOT silently fall
        // back to dialect 1.
        assert_eq!(cset_language_name(9, 9), None);
    }

    /// Build a minimal valid PCM WAV with a caller-supplied number of
    /// `JUNK` chunks, each with the given body size and a fixed
    /// per-chunk fill byte, inserted between `fmt ` and `data`. Mirrors
    /// `wav_with_cset` / `wav_with_ixml` for the filler chunk
    /// documented in Microsoft RIFF MCI §2 "JUNK (Filler) Chunk".
    fn wav_with_junk(junk_sizes: &[usize]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt : 16-byte PCM s16 mono 8000 Hz.
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        for (i, &sz) in junk_sizes.iter().enumerate() {
            buf.extend_from_slice(b"JUNK");
            buf.extend_from_slice(&(sz as u32).to_le_bytes());
            // Fill byte is the chunk index — lets a debugger see which
            // JUNK the bytes belong to without affecting parsing
            // behaviour (the parser must not depend on the contents).
            buf.extend(std::iter::repeat_n(i as u8, sz));
            if sz % 2 == 1 {
                buf.push(0);
            }
        }
        // empty data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// A single 16-byte `JUNK` chunk surfaces its count, total payload
    /// bytes and per-chunk body length under the `wav:junk.*` key
    /// shape. The body contents are not surfaced (the spec defines
    /// them as "no relevant data"). The chunk-walk still locates the
    /// `data` chunk that follows.
    #[test]
    fn junk_single_chunk_surfaces_accounting_metadata() {
        let bytes = wav_with_junk(&[16]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:junk.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:junk.total_bytes"), Some(&"16".to_string()));
        assert_eq!(md.get("wav:junk.0.body_len"), Some(&"16".to_string()));
        // Body contents must not leak into metadata under any key.
        assert!(
            !md.keys()
                .any(|k| k.starts_with("wav:junk") && k.ends_with(".body")),
            "JUNK chunk body must not be surfaced (Microsoft RIFF MCI §2)"
        );
        // The fmt + data path still works end-to-end.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Multiple `JUNK` chunks accumulate into the `count` /
    /// `total_bytes` aggregates and each surfaces its own
    /// `wav:junk.<n>.body_len`. The §2 spec allows arbitrary repetition;
    /// many real writers reserve one slot ahead of `LIST INFO` and a
    /// second ahead of `data` for in-place editing.
    #[test]
    fn junk_multiple_chunks_accumulate() {
        let bytes = wav_with_junk(&[32, 8, 100]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:junk.count"), Some(&"3".to_string()));
        assert_eq!(
            md.get("wav:junk.total_bytes"),
            Some(&(32u64 + 8 + 100).to_string())
        );
        assert_eq!(md.get("wav:junk.0.body_len"), Some(&"32".to_string()));
        assert_eq!(md.get("wav:junk.1.body_len"), Some(&"8".to_string()));
        assert_eq!(md.get("wav:junk.2.body_len"), Some(&"100".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A zero-length `JUNK` chunk is in-range per the §2 "arbitrary
    /// size" language and still increments the count. The
    /// `wav:junk.0.body_len = 0` entry distinguishes "an empty JUNK
    /// was present" from "no JUNK was present at all" — the latter
    /// emits no `wav:junk.*` keys whatsoever.
    #[test]
    fn junk_empty_body_still_counts() {
        let bytes = wav_with_junk(&[0]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:junk.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:junk.total_bytes"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:junk.0.body_len"), Some(&"0".to_string()));
    }

    /// A file with no `JUNK` chunks must not synthesise any
    /// `wav:junk.*` keys (absence is observable: zero `count` keys is
    /// stronger than `count = 0` because it costs no bytes). Baseline
    /// regression guard against a future refactor that initialises the
    /// counter unconditionally.
    #[test]
    fn junk_absent_emits_no_keys() {
        let bytes = wav_with_junk(&[]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(
            !md.keys().any(|k| k.starts_with("wav:junk")),
            "no JUNK chunk → no wav:junk.* metadata"
        );
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// An odd-length `JUNK` body forces a 1-byte word-align pad; the
    /// `data` chunk that follows must still be located correctly
    /// (regression guard matching `ixml_odd_body_padding` /
    /// `cset_odd_body_padding`). RIFF MCI §2 "Chunks" requires all
    /// chunks to be word-aligned with an implicit pad byte when the
    /// body is odd; the pad byte is NOT part of `ckSize`.
    #[test]
    fn junk_odd_body_padding() {
        let bytes = wav_with_junk(&[7]); // odd
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:junk.0.body_len"), Some(&"7".to_string()));
        assert_eq!(md.get("wav:junk.total_bytes"), Some(&"7".to_string()));
        // data chunk located correctly past the pad.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// `JUNK` coexists with `LIST INFO` and `CSET` without disrupting
    /// the rest of the metadata surface. Regression guard for the
    /// chunk-walk ordering JUNK → CSET → LIST(INFO) → JUNK → data
    /// (a realistic shape when a writer reserves filler ahead of both
    /// the metadata block and the audio payload).
    #[test]
    fn junk_coexists_with_other_chunks() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // First JUNK: 12 bytes of filler.
        buf.extend_from_slice(b"JUNK");
        buf.extend_from_slice(&12u32.to_le_bytes());
        buf.extend(std::iter::repeat_n(0xAAu8, 12));
        // CSET: Windows-1252 / USA / US English (canonical 8-byte body).
        let cset = cset_body(1252, 1, 9, 1);
        buf.extend_from_slice(b"CSET");
        buf.extend_from_slice(&(cset.len() as u32).to_le_bytes());
        buf.extend_from_slice(&cset);
        // LIST INFO with INAM = "T" (1 byte + NUL terminator).
        let mut list_body = Vec::new();
        list_body.extend_from_slice(b"INFO");
        list_body.extend_from_slice(b"INAM");
        list_body.extend_from_slice(&1u32.to_le_bytes());
        list_body.extend_from_slice(b"T");
        list_body.push(0);
        buf.extend_from_slice(b"LIST");
        buf.extend_from_slice(&(list_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(&list_body);
        // Second JUNK: 4 bytes of filler ahead of `data`.
        buf.extend_from_slice(b"JUNK");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend(std::iter::repeat_n(0xBBu8, 4));
        // empty data chunk.
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());

        let dmx = open_demux_from_bytes(buf);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        // Both JUNK chunks counted; aggregates reflect both.
        assert_eq!(md.get("wav:junk.count"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:junk.total_bytes"), Some(&"16".to_string()));
        assert_eq!(md.get("wav:junk.0.body_len"), Some(&"12".to_string()));
        assert_eq!(md.get("wav:junk.1.body_len"), Some(&"4".to_string()));
        // Other chunks survived the interleaved JUNK chunks intact.
        assert_eq!(md.get("wav:cset.code_page"), Some(&"1252".to_string()));
        assert_eq!(md.get("title"), Some(&"T".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Build a minimal valid WAV file carrying the supplied top-level
    /// `slnt` (silence) chunks. Each `&[u8]` body is written verbatim as
    /// the chunk payload (canonically a 4-byte LE `dwSamples`, but the
    /// helper lets a test feed a short/long body to exercise the
    /// opaque-body path). `fmt ` is PCM-S16 mono so the demuxer accepts
    /// the file; an empty `data` chunk closes the chunk-walk.
    fn wav_with_slnt(slnt_bodies: &[&[u8]]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        for body in slnt_bodies {
            buf.extend_from_slice(b"slnt");
            buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
            buf.extend_from_slice(body);
            if body.len() % 2 == 1 {
                buf.push(0);
            }
        }
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// A single canonical `slnt` chunk surfaces its `dwSamples` count
    /// under the `wav:slnt.*` accounting keys per Microsoft RIFF MCI §3
    /// "Wave Data". No real silence is synthesised into the decoded
    /// stream; the chunk-walk still locates the `data` chunk that
    /// follows.
    #[test]
    fn slnt_single_chunk_surfaces_sample_count() {
        let bytes = wav_with_slnt(&[&1_000u32.to_le_bytes()]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:slnt.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:slnt.total_samples"), Some(&"1000".to_string()));
        assert_eq!(md.get("wav:slnt.0.samples"), Some(&"1000".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// Multiple top-level `slnt` chunks accumulate into the `count` /
    /// `total_samples` aggregates and each surfaces its own
    /// `wav:slnt.<n>.samples`. The §3 grammar allows the silence chunk
    /// to repeat (the `wavl` alternating-data form); the demuxer
    /// accounts for every occurrence it sees at the top level.
    #[test]
    fn slnt_multiple_chunks_accumulate() {
        let bytes = wav_with_slnt(&[
            &500u32.to_le_bytes(),
            &250u32.to_le_bytes(),
            &44_100u32.to_le_bytes(),
        ]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:slnt.count"), Some(&"3".to_string()));
        assert_eq!(
            md.get("wav:slnt.total_samples"),
            Some(&(500u64 + 250 + 44_100).to_string())
        );
        assert_eq!(md.get("wav:slnt.0.samples"), Some(&"500".to_string()));
        assert_eq!(md.get("wav:slnt.1.samples"), Some(&"250".to_string()));
        assert_eq!(md.get("wav:slnt.2.samples"), Some(&"44100".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `slnt` chunk whose `dwSamples` field is `0` is in-range (a
    /// zero-length silence run) and still increments the count. The
    /// per-chunk `samples = 0` entry distinguishes "an explicit empty
    /// silence run" from "no slnt chunk at all".
    #[test]
    fn slnt_zero_samples_still_counts() {
        let bytes = wav_with_slnt(&[&0u32.to_le_bytes()]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:slnt.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:slnt.total_samples"), Some(&"0".to_string()));
        assert_eq!(md.get("wav:slnt.0.samples"), Some(&"0".to_string()));
    }

    /// A file with no `slnt` chunk must not synthesise any `wav:slnt.*`
    /// keys — absence is observable (zero keys is stronger than
    /// `count = 0`). Regression guard against a future refactor that
    /// initialises the counter unconditionally.
    #[test]
    fn slnt_absent_emits_no_keys() {
        let bytes = wav_with_slnt(&[]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert!(
            !md.keys().any(|k| k.starts_with("wav:slnt")),
            "no slnt chunk → no wav:slnt.* metadata"
        );
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `slnt` body shorter than the 4-byte `dwSamples` field is
    /// treated as opaque: the chunk is still counted (so the reservation
    /// is observable) but contributes nothing to `total_samples` and its
    /// per-chunk `samples` key is omitted. Mirrors how the other
    /// fixed-struct parsers treat an under-length body.
    #[test]
    fn slnt_short_body_is_opaque() {
        // 3-byte body (one short of the 4-byte DWORD) — odd length also
        // exercises the word-align pad on the way to `data`.
        let bytes = wav_with_slnt(&[&[0x01, 0x02, 0x03]]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:slnt.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:slnt.total_samples"), Some(&"0".to_string()));
        assert!(
            !md.contains_key("wav:slnt.0.samples"),
            "under-length slnt body must not surface a samples value"
        );
        // data located correctly past the implicit pad byte.
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `slnt` body longer than the canonical 4 bytes still decodes its
    /// leading `dwSamples` DWORD and tolerates the trailing region for
    /// forward compatibility (matching the §3 forward-extension rule the
    /// `fact` parser follows). An odd over-length body also exercises
    /// the word-align pad ahead of `data`.
    #[test]
    fn slnt_long_body_decodes_leading_dword() {
        // 5-byte body: leading DWORD = 7, one trailing extension byte.
        let mut body = 7u32.to_le_bytes().to_vec();
        body.push(0xFF);
        let bytes = wav_with_slnt(&[&body]);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:slnt.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:slnt.0.samples"), Some(&"7".to_string()));
        assert_eq!(md.get("wav:slnt.total_samples"), Some(&"7".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// `slnt` coexists with `JUNK` and `LIST INFO` without disrupting
    /// the rest of the metadata surface, and the two independent
    /// accounting namespaces (`wav:slnt.*` vs `wav:junk.*`) don't
    /// collide. Regression guard for the chunk-walk ordering
    /// slnt → JUNK → LIST(INFO) → slnt → data.
    #[test]
    fn slnt_coexists_with_other_chunks() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // First slnt: 800 silent samples.
        buf.extend_from_slice(b"slnt");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&800u32.to_le_bytes());
        // JUNK: 6 bytes of filler.
        buf.extend_from_slice(b"JUNK");
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend(std::iter::repeat_n(0xAAu8, 6));
        // LIST INFO with INAM = "T".
        let mut list_body = Vec::new();
        list_body.extend_from_slice(b"INFO");
        list_body.extend_from_slice(b"INAM");
        list_body.extend_from_slice(&1u32.to_le_bytes());
        list_body.extend_from_slice(b"T");
        list_body.push(0);
        buf.extend_from_slice(b"LIST");
        buf.extend_from_slice(&(list_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(&list_body);
        // Second slnt: 200 silent samples.
        buf.extend_from_slice(b"slnt");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&200u32.to_le_bytes());
        // empty data chunk.
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());

        let dmx = open_demux_from_bytes(buf);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:slnt.count"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:slnt.total_samples"), Some(&"1000".to_string()));
        assert_eq!(md.get("wav:slnt.0.samples"), Some(&"800".to_string()));
        assert_eq!(md.get("wav:slnt.1.samples"), Some(&"200".to_string()));
        // JUNK + INFO survived the interleaved slnt chunks intact.
        assert_eq!(md.get("wav:junk.count"), Some(&"1".to_string()));
        assert_eq!(md.get("title"), Some(&"T".to_string()));
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// A `wavl`-form sub-chunk descriptor for the test builder: a 4-byte
    /// FOURCC (`data` / `slnt`) plus a verbatim body. `data` bodies carry
    /// PCM; `slnt` bodies carry the 4-byte `dwSamples` count.
    enum WavlSeg<'a> {
        Data(&'a [u8]),
        Slnt(u32),
    }

    /// Build a minimal valid WAV whose waveform is stored as a
    /// `LIST('wavl' ...)` wave-list (Microsoft RIFF MCI §3 "Storage of
    /// WAVE Data") instead of a top-level `data` chunk. `fmt ` is
    /// PCM-S16 mono; a `fact` chunk carries the authoritative total
    /// sample count (the spec requires `fact` whenever the data lives in
    /// a `wavl` LIST). No top-level `data` chunk is emitted.
    fn wav_with_wavl(segs: &[WavlSeg], fact_samples: u32) -> Vec<u8> {
        // Build the wavl LIST body first so we can size the LIST chunk.
        let mut wavl = Vec::new();
        wavl.extend_from_slice(b"wavl");
        for seg in segs {
            match seg {
                WavlSeg::Data(body) => {
                    wavl.extend_from_slice(b"data");
                    wavl.extend_from_slice(&(body.len() as u32).to_le_bytes());
                    wavl.extend_from_slice(body);
                    if body.len() % 2 == 1 {
                        wavl.push(0);
                    }
                }
                WavlSeg::Slnt(samples) => {
                    wavl.extend_from_slice(b"slnt");
                    wavl.extend_from_slice(&4u32.to_le_bytes());
                    wavl.extend_from_slice(&samples.to_le_bytes());
                }
            }
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // fact chunk — required for wavl-form data.
        buf.extend_from_slice(b"fact");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&fact_samples.to_le_bytes());
        // The wavl LIST itself.
        buf.extend_from_slice(b"LIST");
        buf.extend_from_slice(&(wavl.len() as u32).to_le_bytes());
        buf.extend_from_slice(&wavl);
        buf
    }

    /// A single-`data`-segment `wavl` LIST is decodable: the demuxer
    /// anchors the cursor at the embedded `data` payload and yields it
    /// byte-for-byte, exactly as a top-level `data` chunk would.
    #[test]
    fn wavl_single_data_segment_decodes() {
        let pcm: Vec<u8> = (0..40u8).collect();
        let bytes = wav_with_wavl(&[WavlSeg::Data(&pcm)], 20);
        let mut dmx = open_demux_from_bytes(bytes);
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, pcm);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:wavl.segment_count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:wavl.data_count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:wavl.data_bytes"), Some(&"40".to_string()));
        assert_eq!(md.get("wav:wavl.0.kind"), Some(&"data".to_string()));
        assert_eq!(md.get("wav:wavl.0.length"), Some(&"40".to_string()));
    }

    /// A `data`/`slnt`/`data` `wavl` LIST surfaces every segment, anchors
    /// the decode cursor at the FIRST `data` segment, and routes the
    /// embedded `slnt` through the shared `wav:slnt.*` accounting so the
    /// silent-sample total matches a top-level-`slnt` file. The `fact`
    /// chunk is the authoritative duration (data_size/block_align is
    /// meaningless for a segmented waveform).
    #[test]
    fn wavl_interleaved_data_silence_surfaces_segments() {
        let a: Vec<u8> = (0..8u8).collect();
        let b: Vec<u8> = (100..108u8).collect();
        let bytes = wav_with_wavl(
            &[WavlSeg::Data(&a), WavlSeg::Slnt(500), WavlSeg::Data(&b)],
            1000,
        );
        let mut dmx = open_demux_from_bytes(bytes);
        // First data segment is the decode anchor.
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, a);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:wavl.segment_count"), Some(&"3".to_string()));
        assert_eq!(md.get("wav:wavl.data_count"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:wavl.data_bytes"), Some(&"16".to_string()));
        assert_eq!(md.get("wav:wavl.0.kind"), Some(&"data".to_string()));
        assert_eq!(md.get("wav:wavl.1.kind"), Some(&"slnt".to_string()));
        assert_eq!(md.get("wav:wavl.2.kind"), Some(&"data".to_string()));
        // Embedded slnt feeds the shared silence accounting.
        assert_eq!(md.get("wav:slnt.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:slnt.total_samples"), Some(&"500".to_string()));
        assert_eq!(md.get("wav:slnt.0.samples"), Some(&"500".to_string()));
        // fact-derived duration: 1000 samples at 8000 Hz.
        let s = &dmx.streams()[0];
        assert_eq!(s.duration, Some(1000));
    }

    /// A silence-only `wavl` LIST (no `data` segment) carries no
    /// decodable audio. The §3 grammar permits it; the demuxer must
    /// reject the file as having no waveform rather than panicking, while
    /// still having surfaced the segment metadata it walked.
    #[test]
    fn wavl_silence_only_has_no_waveform() {
        let bytes = wav_with_wavl(&[WavlSeg::Slnt(1000), WavlSeg::Slnt(2000)], 3000);
        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
        match open_demuxer(rs, &oxideav_core::NullCodecResolver) {
            Ok(_) => panic!("silence-only wavl must be rejected as having no waveform"),
            Err(Error::InvalidData(_)) => {}
            Err(e) => panic!("expected InvalidData, got {e:?}"),
        }
    }

    /// An odd-length `data` segment inside a `wavl` LIST forces a 1-byte
    /// word-align pad; a following segment must still be located. RIFF
    /// MCI §2 word-alignment applies to sub-chunks inside a LIST too.
    #[test]
    fn wavl_odd_data_segment_padding() {
        let a: Vec<u8> = (0..7u8).collect(); // odd length → 1 pad byte
        let b: Vec<u8> = (50..54u8).collect();
        let bytes = wav_with_wavl(&[WavlSeg::Data(&a), WavlSeg::Data(&b)], 100);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:wavl.segment_count"), Some(&"2".to_string()));
        assert_eq!(md.get("wav:wavl.0.length"), Some(&"7".to_string()));
        assert_eq!(md.get("wav:wavl.1.kind"), Some(&"data".to_string()));
        assert_eq!(md.get("wav:wavl.1.length"), Some(&"4".to_string()));
    }

    /// Locate the first chunk with the given 4-byte FOURCC in a
    /// freshly-muxed WAV file. Helper for the muxer-side `fact`
    /// presence/absence assertions above — uses a naive linear scan
    /// rather than walking the RIFF tree because the tests only need
    /// "is this FOURCC anywhere" not "is it at the right depth".
    /// Conservative on overlap (the FOURCC could appear inside an
    /// `INFO` text payload) — caller chooses test inputs that avoid
    /// false positives.
    fn find_chunk(buf: &[u8], fourcc: &[u8; 4]) -> Option<usize> {
        // Skip the 12-byte RIFF/WAVE header.
        let mut i = 12usize;
        while i + 8 <= buf.len() {
            if &buf[i..i + 4] == fourcc {
                return Some(i);
            }
            let sz = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
            i += 8 + sz + (sz % 2);
        }
        None
    }

    /// `fmt_guid` produces the canonical text form: first three groups
    /// little-endian, trailing two big-endian. This is the format
    /// `mmreg.h` definitions use (e.g. `KSDATAFORMAT_SUBTYPE_PCM` =
    /// `00000001-0000-0010-8000-00AA00389B71`).
    #[test]
    fn guid_canonical_text() {
        assert_eq!(fmt_guid(&GUID_PCM), "00000001-0000-0010-8000-00AA00389B71");
        assert_eq!(
            fmt_guid(&GUID_IEEE_FLOAT),
            "00000003-0000-0010-8000-00AA00389B71"
        );
        assert_eq!(fmt_guid(&GUID_ALAW), "00000006-0000-0010-8000-00AA00389B71");
        assert_eq!(
            fmt_guid(&GUID_MULAW),
            "00000007-0000-0010-8000-00AA00389B71"
        );
    }

    /// Build a one-entry `LIST INFO` sub-chunk header + payload for
    /// `id` carrying the ASCII text `text` plus a single NUL terminator.
    /// Used by the §3 baseline-coverage tests below to feed one INFO
    /// sub-ID at a time through the demuxer.
    fn info_subchunk(id: &[u8; 4], text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(id);
        // Payload = text bytes + 1 NUL terminator (the §3 ZSTR form).
        let payload_len = text.len() + 1;
        out.extend_from_slice(&(payload_len as u32).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
        out.push(0);
        // Word-align the sub-chunk to the next even boundary, per the
        // RIFF padding rule (§2 "Chunks": "Padding, if present, is not
        // included in `ckSize`.").
        if payload_len % 2 == 1 {
            out.push(0);
        }
        out
    }

    /// Build a minimal valid WAV file whose `LIST INFO` chunk carries
    /// the supplied per-sub-ID payloads. Empty `data` keeps the file
    /// short; `fmt ` is PCM-S16 mono so the demuxer accepts it.
    fn wav_with_info_entries(entries: &[(&[u8; 4], &str)]) -> Vec<u8> {
        let mut list_body = Vec::new();
        list_body.extend_from_slice(b"INFO");
        for (id, text) in entries {
            list_body.extend_from_slice(&info_subchunk(id, text));
        }
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&FMT_PCM.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8_000u32.to_le_bytes());
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"LIST");
        buf.extend_from_slice(&(list_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(&list_body);
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// Every Microsoft RIFF MCI §3 INFO List Chunk baseline sub-ID (the
    /// 23 entries on pp. 2-14 to 2-16) round-trips through
    /// `parse_info_list` to its conventional key name. Spec text per
    /// `docs/container/riff/metadata/microsoft-riffmci.pdf`.
    #[test]
    fn info_list_full_baseline_round_trip() {
        // One representative payload per sub-ID, mirroring the §3
        // example phrasing where the spec gives one.
        let entries: &[(&[u8; 4], &str)] = &[
            (b"IARL", "Library of Congress"),
            (b"IART", "Michaelangelo"),
            (b"ICMS", "Pope Julian II"),
            (b"ICMT", "General comment."),
            (b"ICOP", "Copyright Encyclopedia International 1991"),
            (b"ICRD", "1553-05-03"),
            (b"ICRP", "lower right corner"),
            (b"IDIM", "8.5 in h, 11 in w"),
            (b"IDPI", "300"),
            (b"IENG", "Smith, John; Adams, Joe"),
            (b"IGNR", "landscape"),
            (b"IKEY", "Seattle; aerial view; scenery"),
            (b"ILGT", "+10"),
            (b"IMED", "lithograph"),
            (b"INAM", "Seattle From Above"),
            (b"IPLT", "256"),
            (b"IPRD", "Encyclopedia of Pacific Northwest Geography"),
            (b"ISBJ", "Aerial view of Seattle"),
            (b"ISFT", "Microsoft WaveEdit"),
            (b"ISHP", "+5"),
            (b"ISRC", "Trey Research"),
            (b"ISRF", "slide"),
            (b"ITCH", "Smith, John"),
        ];

        let bytes = wav_with_info_entries(entries);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        // The expected key for each baseline sub-ID. Order matches
        // `entries` so a failure points at the offending FOURCC.
        let expected: &[(&str, &str)] = &[
            ("archival_location", "Library of Congress"),
            ("artist", "Michaelangelo"),
            ("commissioned", "Pope Julian II"),
            ("comment", "General comment."),
            ("copyright", "Copyright Encyclopedia International 1991"),
            ("date", "1553-05-03"),
            ("cropped", "lower right corner"),
            ("dimensions", "8.5 in h, 11 in w"),
            ("dpi", "300"),
            ("engineer", "Smith, John; Adams, Joe"),
            ("genre", "landscape"),
            ("keywords", "Seattle; aerial view; scenery"),
            ("lightness", "+10"),
            ("medium", "lithograph"),
            ("title", "Seattle From Above"),
            ("palette_setting", "256"),
            ("album", "Encyclopedia of Pacific Northwest Geography"),
            ("subject", "Aerial view of Seattle"),
            ("encoder", "Microsoft WaveEdit"),
            ("sharpness", "+5"),
            ("source", "Trey Research"),
            ("source_form", "slide"),
            ("technician", "Smith, John"),
        ];
        for (key, value) in expected {
            assert_eq!(
                md.get(*key),
                Some(&(*value).to_string()),
                "INFO key {key:?} missing or wrong; got {:?}",
                md.get(*key)
            );
        }
    }

    /// Each pre-r221 INFO sub-ID still maps to the same conventional
    /// key — the §3 baseline expansion does not perturb the four
    /// widely-quoted audio-tag aliases (`title`, `artist`, `album`,
    /// `comment`, `genre`, `copyright`, `engineer`, `technician`,
    /// `encoder`, `subject`, `date`) or the `track` extension.
    #[test]
    fn info_id_to_key_legacy_aliases_preserved() {
        assert_eq!(info_id_to_key(b"INAM"), Some("title"));
        assert_eq!(info_id_to_key(b"IART"), Some("artist"));
        assert_eq!(info_id_to_key(b"IPRD"), Some("album"));
        assert_eq!(info_id_to_key(b"ICMT"), Some("comment"));
        assert_eq!(info_id_to_key(b"ICRD"), Some("date"));
        assert_eq!(info_id_to_key(b"IGNR"), Some("genre"));
        assert_eq!(info_id_to_key(b"ICOP"), Some("copyright"));
        assert_eq!(info_id_to_key(b"IENG"), Some("engineer"));
        assert_eq!(info_id_to_key(b"ITCH"), Some("technician"));
        assert_eq!(info_id_to_key(b"ISFT"), Some("encoder"));
        assert_eq!(info_id_to_key(b"ISBJ"), Some("subject"));
        assert_eq!(info_id_to_key(b"ITRK"), Some("track"));
    }

    /// Every §3 baseline sub-ID resolves to `Some` and every unknown
    /// FOURCC resolves to `None`. The `Some` branch is sufficient for
    /// the FOURCC-typo regression guard the round-trip test does not
    /// catch (a misspelt `b"IRAL"` for `IARL` would silently lose
    /// data otherwise).
    #[test]
    fn info_id_to_key_baseline_completeness() {
        const BASELINE: &[&[u8; 4]] = &[
            b"IARL", b"IART", b"ICMS", b"ICMT", b"ICOP", b"ICRD", b"ICRP", b"IDIM", b"IDPI",
            b"IENG", b"IGNR", b"IKEY", b"ILGT", b"IMED", b"INAM", b"IPLT", b"IPRD", b"ISBJ",
            b"ISFT", b"ISHP", b"ISRC", b"ISRF", b"ITCH",
        ];
        for id in BASELINE {
            assert!(
                info_id_to_key(id).is_some(),
                "baseline INFO sub-ID {:?} missing from info_id_to_key",
                std::str::from_utf8(*id).unwrap()
            );
        }
        // Negative: a definitely-unregistered FOURCC.
        assert_eq!(info_id_to_key(b"ZZZZ"), None);
        // Negative: easy typo of IARL.
        assert_eq!(info_id_to_key(b"IRAL"), None);
    }

    /// Every extended `INFO` sub-ID from ExifTool's RIFF Info Tags
    /// table (`docs/container/riff/metadata/exiftool-riff-tags.html`)
    /// resolves to its documented snake_case key. The mapping covers
    /// the per-stream audio-language slots (`IAS1`..`IAS9`), the
    /// Windows-Media "more info" set, and the common production-credit
    /// tags — each a plain ZSTR field surfaced exactly like a baseline
    /// entry. End-to-end through the demuxer so the `parse_info_list`
    /// path is exercised, not just the lookup table.
    #[test]
    fn info_list_extended_subids_round_trip() {
        let entries: &[(&[u8; 4], &str)] = &[
            (b"IAS1", "English"),
            (b"IAS2", "French"),
            (b"IAS3", "German"),
            (b"IAS4", "Spanish"),
            (b"IAS5", "Italian"),
            (b"IAS6", "Dutch"),
            (b"IAS7", "Polish"),
            (b"IAS8", "Czech"),
            (b"IAS9", "Greek"),
            (b"IBSU", "https://example.com/"),
            (b"ICAS", "2"),
            (b"ICDS", "Edith Head"),
            (b"ICNM", "Gregg Toland"),
            (b"ICNT", "US"),
            (b"IDIT", "2026:06:16 12:00:00"),
            (b"IDST", "Acme Distribution"),
            (b"IEDT", "Walter Murch"),
            (b"IENC", "FooEncoder 1.0"),
            (b"ILGU", "https://example.com/logo"),
            (b"ILIU", "https://example.com/icon.png"),
            (b"ILNG", "eng"),
            (b"IMBI", "banner.png"),
            (b"IMBU", "https://example.com/banner"),
            (b"IMIT", "See website for details"),
            (b"IMIU", "https://example.com/info"),
            (b"IMUS", "John Williams"),
            (b"IPDS", "Hermann Warm"),
            (b"IPRO", "Jane Producer"),
            (b"IRIP", "RipTool"),
            (b"IRTD", "5"),
            (b"ISGN", "Documentary"),
            (b"ISMP", "01:23:45:12"),
            (b"ISTD", "Pacific Studios"),
            (b"ISTR", "A. Actor, B. Actor"),
            (b"IWMU", "https://example.com/wm"),
            (b"IWRI", "C. Writer"),
        ];

        let bytes = wav_with_info_entries(entries);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        let expected: &[(&str, &str)] = &[
            ("first_language", "English"),
            ("second_language", "French"),
            ("third_language", "German"),
            ("fourth_language", "Spanish"),
            ("fifth_language", "Italian"),
            ("sixth_language", "Dutch"),
            ("seventh_language", "Polish"),
            ("eighth_language", "Czech"),
            ("ninth_language", "Greek"),
            ("base_url", "https://example.com/"),
            ("default_audio_stream", "2"),
            ("costume_designer", "Edith Head"),
            ("cinematographer", "Gregg Toland"),
            ("country", "US"),
            ("date_time_original", "2026:06:16 12:00:00"),
            ("distributed_by", "Acme Distribution"),
            ("edited_by", "Walter Murch"),
            ("encoded_by", "FooEncoder 1.0"),
            ("logo_url", "https://example.com/logo"),
            ("logo_icon_url", "https://example.com/icon.png"),
            ("language", "eng"),
            ("more_info_banner_image", "banner.png"),
            ("more_info_banner_url", "https://example.com/banner"),
            ("more_info_text", "See website for details"),
            ("more_info_url", "https://example.com/info"),
            ("music_by", "John Williams"),
            ("production_designer", "Hermann Warm"),
            ("produced_by", "Jane Producer"),
            ("ripped_by", "RipTool"),
            ("rating", "5"),
            ("secondary_genre", "Documentary"),
            ("time_code", "01:23:45:12"),
            ("production_studio", "Pacific Studios"),
            ("starring", "A. Actor, B. Actor"),
            ("watermark_url", "https://example.com/wm"),
            ("written_by", "C. Writer"),
        ];
        for (key, value) in expected {
            assert_eq!(
                md.get(*key),
                Some(&(*value).to_string()),
                "extended INFO key {key:?} missing or wrong; got {:?}",
                md.get(*key)
            );
        }
    }

    /// The extended `INFO` sub-IDs do not collide with the baseline
    /// group: each extended FOURCC resolves to a `Some` key distinct
    /// from every baseline key, and a representative non-`I` FOURCC
    /// still returns `None`.
    #[test]
    fn info_id_to_key_extended_completeness() {
        const EXTENDED: &[&[u8; 4]] = &[
            b"IAS1", b"IAS2", b"IAS3", b"IAS4", b"IAS5", b"IAS6", b"IAS7", b"IAS8", b"IAS9",
            b"IBSU", b"ICAS", b"ICDS", b"ICNM", b"ICNT", b"IDIT", b"IDST", b"IEDT", b"IENC",
            b"ILGU", b"ILIU", b"ILNG", b"IMBI", b"IMBU", b"IMIT", b"IMIU", b"IMUS", b"IPDS",
            b"IPRO", b"IRIP", b"IRTD", b"ISGN", b"ISMP", b"ISTD", b"ISTR", b"IWMU", b"IWRI",
        ];
        let mut seen = std::collections::HashSet::new();
        for id in EXTENDED {
            let key = info_id_to_key(id).unwrap_or_else(|| {
                panic!(
                    "extended INFO sub-ID {:?} missing from info_id_to_key",
                    std::str::from_utf8(*id).unwrap()
                )
            });
            assert!(
                seen.insert(key),
                "extended INFO key {key:?} collides with another sub-ID"
            );
        }
    }

    /// A `LIST INFO` chunk that contains an unknown sub-ID alongside
    /// known ones — the unknown sub-ID is skipped without disturbing
    /// the known ones' surfacing. Regression guard for the
    /// "skip-unknown" branch in `parse_info_list`.
    #[test]
    fn info_list_unknown_subchunk_is_skipped() {
        let entries: &[(&[u8; 4], &str)] = &[
            (b"INAM", "Title"),
            // `IZZZ` is not in the §3 baseline; the parser drops its
            // bytes silently.
            (b"IZZZ", "ignored"),
            (b"IART", "Artist"),
        ];
        let bytes = wav_with_info_entries(entries);
        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("title"), Some(&"Title".to_string()));
        assert_eq!(md.get("artist"), Some(&"Artist".to_string()));
        // No synthetic key created for the unknown FOURCC.
        assert!(md
            .keys()
            .all(|k| !k.contains("IZZZ") && !k.contains("izzz")));
    }

    // -------- RF64 / BW64 (EBU Tech 3306 / ITU-R BS.2088) ---------------

    /// Build a synthetic RF64-or-BW64 WAV from in-memory parts:
    /// `magic` is `b"RF64"` or `b"BW64"`, the legacy 32-bit RIFF /
    /// data / fact size fields are forced to the `0xFFFFFFFF`
    /// sentinel per EBU Tech 3306 §3, and `ds64` carries the
    /// authoritative 64-bit sizes ahead of `fmt `/`fact`/`data`.
    /// `extra_table` is the optional table of per-chunk-ID 64-bit
    /// overrides surfaced after the three mandatory ds64 fields.
    #[allow(clippy::too_many_arguments)]
    fn synth_rf64(
        magic: &[u8; 4],
        ds64_data_size: u64,
        ds64_sample_count: u64,
        extra_table: &[([u8; 4], u64)],
        data_payload: &[u8],
        channels: u16,
        sample_rate: u32,
        bits_per_sample: u16,
    ) -> Vec<u8> {
        // Build `ds64` body first so we can compute its size.
        let mut ds64_body: Vec<u8> = Vec::new();
        // riffSize will be back-patched once we know the full file
        // size. We push a placeholder here.
        ds64_body.extend_from_slice(&0u64.to_le_bytes());
        ds64_body.extend_from_slice(&ds64_data_size.to_le_bytes());
        ds64_body.extend_from_slice(&ds64_sample_count.to_le_bytes());
        ds64_body.extend_from_slice(&(extra_table.len() as u32).to_le_bytes());
        for (id, sz) in extra_table {
            ds64_body.extend_from_slice(id);
            ds64_body.extend_from_slice(&sz.to_le_bytes());
        }

        // `fmt ` chunk: legacy 16-byte WAVEFORMAT for PCM (the spec's
        // standard ds64 example targets PCM streams).
        let block_align = (bits_per_sample / 8) * channels;
        let byte_rate = sample_rate * block_align as u32;
        let mut fmt_body: Vec<u8> = Vec::new();
        fmt_body.extend_from_slice(&WAVE_FORMAT_PCM.to_le_bytes());
        fmt_body.extend_from_slice(&channels.to_le_bytes());
        fmt_body.extend_from_slice(&sample_rate.to_le_bytes());
        fmt_body.extend_from_slice(&byte_rate.to_le_bytes());
        fmt_body.extend_from_slice(&block_align.to_le_bytes());
        fmt_body.extend_from_slice(&bits_per_sample.to_le_bytes());

        // `fact` chunk: legacy 4-byte dwFileSize forced to the
        // sentinel so the demuxer must consult ds64.
        let fact_body = SIZE64_SENTINEL.to_le_bytes();

        // Assemble file: magic + sentinel + "WAVE" + ds64 + fmt +
        // fact + data + payload.
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(magic);
        out.extend_from_slice(&SIZE64_SENTINEL.to_le_bytes());
        out.extend_from_slice(b"WAVE");

        out.extend_from_slice(b"ds64");
        out.extend_from_slice(&(ds64_body.len() as u32).to_le_bytes());
        let ds64_riff_field_off = out.len();
        out.extend_from_slice(&ds64_body);

        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&(fmt_body.len() as u32).to_le_bytes());
        out.extend_from_slice(&fmt_body);

        out.extend_from_slice(b"fact");
        out.extend_from_slice(&(fact_body.len() as u32).to_le_bytes());
        out.extend_from_slice(&fact_body);

        out.extend_from_slice(b"data");
        out.extend_from_slice(&SIZE64_SENTINEL.to_le_bytes());
        out.extend_from_slice(data_payload);

        // Back-patch the placeholder ds64.riffSize field with the
        // real total minus 8 (matching the legacy 32-bit field's
        // semantics — file size minus the 8-byte form header).
        let riff_size_64 = (out.len() as u64) - 8;
        out[ds64_riff_field_off..ds64_riff_field_off + 8]
            .copy_from_slice(&riff_size_64.to_le_bytes());
        out
    }

    /// RF64 PCM S16 mono file with `ds64` carrying the 64-bit
    /// data-size and sample-count overrides. Demuxer must read the
    /// payload through the ds64 path, surface
    /// `wav:rf64.magic = "RF64"` plus the three 64-bit fields, and
    /// report the correct per-channel sample count.
    #[test]
    fn rf64_demux_pcm_s16_with_ds64_overrides() {
        // 100 samples × 2 bytes = 200 bytes of payload. Small but
        // exercises every ds64-promotion branch.
        let payload: Vec<u8> = (0..200u8).collect();
        let bytes = synth_rf64(b"RF64", 200, 100, &[], &payload, 1, 48_000, 16);

        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        assert_eq!(md.get("wav:rf64.magic"), Some(&"RF64".to_string()));
        assert_eq!(md.get("wav:rf64.data_size"), Some(&"200".to_string()));
        assert_eq!(md.get("wav:rf64.sample_count"), Some(&"100".to_string()));
        assert_eq!(md.get("wav:rf64.table.count"), Some(&"0".to_string()));

        // The duration was driven from the ds64-promoted fact field,
        // not the 4-byte legacy value (which was the sentinel).
        let stream = &dmx.streams()[0];
        assert_eq!(stream.duration, Some(100));
        assert_eq!(stream.params.codec_id, CodecId::new("pcm_s16le"));
    }

    /// BW64 magic (instead of RF64) is the ADM-carrying form per
    /// ITU-R BS.2088 — same ds64 layout, different top-level FOURCC.
    /// The demuxer must accept it identically.
    #[test]
    fn bw64_demux_pcm_s16_treated_as_rf64() {
        let payload: Vec<u8> = (0..40u8).collect();
        let bytes = synth_rf64(b"BW64", 40, 20, &[], &payload, 1, 48_000, 16);

        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:rf64.magic"), Some(&"BW64".to_string()));
        assert_eq!(md.get("wav:rf64.data_size"), Some(&"40".to_string()));
        assert_eq!(md.get("wav:rf64.sample_count"), Some(&"20".to_string()));
        assert_eq!(dmx.streams()[0].duration, Some(20));
    }

    /// A non-`data` chunk that carries the 32-bit sentinel must be
    /// resolved via the `ds64.table` lookup. Build a file with a
    /// `LIST INFO` chunk whose on-wire size is the sentinel and
    /// whose real size is recorded in the table.
    #[test]
    fn rf64_table_promotes_non_data_chunk_size() {
        // Construct a `LIST INFO INAM "Hello"` chunk body manually.
        let mut list_body: Vec<u8> = Vec::new();
        list_body.extend_from_slice(b"INFO");
        list_body.extend_from_slice(b"INAM");
        list_body.extend_from_slice(&6u32.to_le_bytes()); // ZSTR "Hello\0"
        list_body.extend_from_slice(b"Hello\0");
        let list_body_len = list_body.len() as u64;

        // Build a file with: magic + sentinel + WAVE + ds64(table =
        // [(LIST, list_body_len)]) + LIST (with sentinel size) +
        // fmt + data.
        let mut ds64_body: Vec<u8> = Vec::new();
        ds64_body.extend_from_slice(&0u64.to_le_bytes()); // riffSize placeholder
        ds64_body.extend_from_slice(&8u64.to_le_bytes()); // dataSize = 8
        ds64_body.extend_from_slice(&4u64.to_le_bytes()); // sampleCount = 4
        ds64_body.extend_from_slice(&1u32.to_le_bytes()); // tableLength
        ds64_body.extend_from_slice(b"LIST");
        ds64_body.extend_from_slice(&list_body_len.to_le_bytes());

        let payload: Vec<u8> = (0..8u8).collect();
        let fmt_body = {
            let mut f = Vec::new();
            f.extend_from_slice(&WAVE_FORMAT_PCM.to_le_bytes());
            f.extend_from_slice(&1u16.to_le_bytes()); // mono
            f.extend_from_slice(&48_000u32.to_le_bytes());
            f.extend_from_slice(&(48_000u32 * 2).to_le_bytes());
            f.extend_from_slice(&2u16.to_le_bytes()); // block_align
            f.extend_from_slice(&16u16.to_le_bytes());
            f
        };

        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"RF64");
        out.extend_from_slice(&SIZE64_SENTINEL.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"ds64");
        out.extend_from_slice(&(ds64_body.len() as u32).to_le_bytes());
        let ds64_riff_off = out.len();
        out.extend_from_slice(&ds64_body);
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&(fmt_body.len() as u32).to_le_bytes());
        out.extend_from_slice(&fmt_body);
        // LIST with the sentinel — real size comes from ds64.table.
        out.extend_from_slice(b"LIST");
        out.extend_from_slice(&SIZE64_SENTINEL.to_le_bytes());
        out.extend_from_slice(&list_body);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&SIZE64_SENTINEL.to_le_bytes());
        out.extend_from_slice(&payload);
        let riff_size_64 = (out.len() as u64) - 8;
        out[ds64_riff_off..ds64_riff_off + 8].copy_from_slice(&riff_size_64.to_le_bytes());

        let dmx = open_demux_from_bytes(out);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();

        // The ds64 table fields surfaced.
        assert_eq!(md.get("wav:rf64.table.count"), Some(&"1".to_string()));
        assert_eq!(md.get("wav:rf64.table.0.id"), Some(&"LIST".to_string()));
        assert_eq!(
            md.get("wav:rf64.table.0.size"),
            Some(&list_body_len.to_string())
        );
        // And the LIST INFO sub-chunk inside it was still parsed,
        // proving the sentinel resolution worked and the chunk-walk
        // skipped the right number of bytes to find `data`.
        assert_eq!(md.get("title"), Some(&"Hello".to_string()));
    }

    /// A plain `RIFF` file with a `data` chunk that carries the
    /// `0xFFFFFFFF` sentinel but no `ds64` chunk is malformed under
    /// EBU Tech 3306 §3 — the sentinel demands a ds64 override. The
    /// demuxer must reject this rather than silently misreading the
    /// data size as 4 GiB.
    #[test]
    fn rf64_sentinel_without_ds64_is_rejected() {
        // Minimal `RIFF`/WAVE file with the sentinel in the data
        // size and no ds64 chunk.
        let fmt_body = {
            let mut f = Vec::new();
            f.extend_from_slice(&WAVE_FORMAT_PCM.to_le_bytes());
            f.extend_from_slice(&1u16.to_le_bytes());
            f.extend_from_slice(&48_000u32.to_le_bytes());
            f.extend_from_slice(&(48_000u32 * 2).to_le_bytes());
            f.extend_from_slice(&2u16.to_le_bytes());
            f.extend_from_slice(&16u16.to_le_bytes());
            f
        };
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&(fmt_body.len() as u32).to_le_bytes());
        out.extend_from_slice(&fmt_body);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&SIZE64_SENTINEL.to_le_bytes());

        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(out));
        let res = open_demuxer(rs, &oxideav_core::NullCodecResolver);
        assert!(res.is_err(), "sentinel without ds64 must be rejected");
    }

    /// A ds64 chunk body shorter than the 28-byte mandatory prefix
    /// is malformed under EBU Tech 3306 Annex A.2 — the three
    /// `int64` fields plus the `tableLength` `int32` add up to
    /// exactly 28 bytes regardless of whether the table is empty.
    #[test]
    fn rf64_ds64_short_body_is_rejected() {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"RF64");
        out.extend_from_slice(&SIZE64_SENTINEL.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"ds64");
        out.extend_from_slice(&8u32.to_le_bytes()); // way too short
        out.extend_from_slice(&[0u8; 8]);

        use std::io::Cursor;
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(out));
        let res = open_demuxer(rs, &oxideav_core::NullCodecResolver);
        assert!(res.is_err(), "ds64 < 28 bytes must be rejected");
    }

    // --- RF64 / BW64 write side --------------------------------------------

    /// `Rf64Mode::Force` always emits the 64-bit form even for a tiny
    /// payload: the magic is `RF64`, the legacy RIFF/`data` size fields
    /// carry the `0xFFFFFFFF` sentinel, and a `ds64` chunk holding the
    /// real sizes is the first chunk after `WAVE`
    /// (EBU Tech 3306 v2 §3 / ITU-R BS.2088-2 §4).
    #[test]
    fn rf64_force_small_file_emits_ds64() {
        let payload = vec![0u8; 4 * 100]; // 100 stereo s16 frames
        let stream = make_stream(SampleFormat::S16, 2, 48_000);
        let opts = WavMuxOptions::default().with_rf64(Rf64Mode::Force);
        let bytes = mux_to_bytes(&stream, &payload, opts, "rf64-force");

        // Top-level magic flipped to RF64; legacy RIFF size = sentinel.
        assert_eq!(&bytes[0..4], b"RF64");
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            SIZE64_SENTINEL
        );
        assert_eq!(&bytes[8..12], b"WAVE");

        // ds64 is the first chunk after WAVE, 28-byte body.
        assert_eq!(&bytes[12..16], b"ds64");
        assert_eq!(
            u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            DS64_FIXED_BODY_LEN
        );
        let body = &bytes[20..20 + DS64_FIXED_BODY_LEN as usize];
        let riff_size = u64::from_le_bytes(body[0..8].try_into().unwrap());
        let data_size = u64::from_le_bytes(body[8..16].try_into().unwrap());
        let sample_count = u64::from_le_bytes(body[16..24].try_into().unwrap());
        let table_len = u32::from_le_bytes(body[24..28].try_into().unwrap());
        assert_eq!(data_size, payload.len() as u64);
        // 100 frames * 2ch * 2 bytes/sample → 100 per-channel samples.
        assert_eq!(sample_count, 100);
        assert_eq!(riff_size, bytes.len() as u64 - 8);
        assert_eq!(table_len, 0);
    }

    /// A `Force`-mode file round-trips through the demuxer: the ds64
    /// sizes resolve the sentinel-tagged `data` chunk, and the metadata
    /// surfaces `wav:rf64.magic == RF64` with the same 64-bit sizes.
    #[test]
    fn rf64_force_round_trips() {
        let samples: Vec<i16> = (0..500).map(|i| ((i * 17) - 4000) as i16).collect();
        let mut payload = Vec::with_capacity(samples.len() * 2);
        for s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }
        let stream = make_stream(SampleFormat::S16, 1, 44_100);
        let opts = WavMuxOptions::default().with_rf64(Rf64Mode::Force);
        let bytes = mux_to_bytes(&stream, &payload, opts, "rf64-force-rt");

        let mut dmx = open_demux_from_bytes(bytes.clone());
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:rf64.magic"), Some(&"RF64".to_string()));
        assert_eq!(
            md.get("wav:rf64.data_size"),
            Some(&payload.len().to_string())
        );
        // The full payload comes back byte-identical despite the
        // sentinel in the 32-bit `data` size field.
        let mut out_bytes = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out_bytes.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out_bytes, payload);
    }

    /// When an ADM `chna` chunk is present the forced 64-bit magic is
    /// `BW64` (ITU-R BS.2088) rather than `RF64` (EBU Tech 3306).
    #[test]
    fn rf64_force_with_chna_is_bw64() {
        let payload = vec![0u8; 2 * 50];
        let stream = make_stream(SampleFormat::S16, 1, 48_000);
        let chna = ChnaChunk {
            num_tracks: 0,
            num_uids: 0,
            ids: Vec::new(),
        };
        let opts = WavMuxOptions::default()
            .with_chna(chna)
            .with_rf64(Rf64Mode::Force);
        let bytes = mux_to_bytes(&stream, &payload, opts, "bw64-force-chna");
        assert_eq!(&bytes[0..4], b"BW64");

        let dmx = open_demux_from_bytes(bytes);
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:rf64.magic"), Some(&"BW64".to_string()));
    }

    /// `Rf64Mode::Reserve` writes a `ds64`-sized `JUNK` placeholder up
    /// front. For a small file that never overflows 32 bits the magic
    /// stays `RIFF`, the placeholder is left as an inert `JUNK` chunk,
    /// and the file is a perfectly ordinary WAVE that the demuxer reads
    /// (the `JUNK` accounting keys appear). BS.2088-2 §3.6.
    #[test]
    fn rf64_reserve_small_file_stays_riff() {
        let samples: Vec<i16> = (0..300).map(|i| (i * 11 - 1500) as i16).collect();
        let mut payload = Vec::with_capacity(samples.len() * 2);
        for s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }
        let stream = make_stream(SampleFormat::S16, 1, 48_000);
        let opts = WavMuxOptions::default().with_rf64(Rf64Mode::Reserve);
        let bytes = mux_to_bytes(&stream, &payload, opts, "rf64-reserve");

        // Magic stays RIFF; placeholder is an inert JUNK chunk after WAVE.
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"JUNK");
        assert_eq!(
            u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            DS64_FIXED_BODY_LEN
        );

        // Demuxer reads it as a normal file; the JUNK accounting keys
        // show the placeholder, and the payload round-trips.
        let mut dmx = open_demux_from_bytes(bytes.clone());
        let md: std::collections::HashMap<String, String> =
            dmx.metadata().iter().cloned().collect();
        assert_eq!(md.get("wav:junk.count"), Some(&"1".to_string()));
        assert_eq!(
            md.get("wav:junk.total_bytes"),
            Some(&DS64_FIXED_BODY_LEN.to_string())
        );
        // No RF64 keys — this is a plain RIFF file.
        assert_eq!(md.get("wav:rf64.magic"), None);
        let mut out_bytes = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out_bytes.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out_bytes, payload);
    }

    /// The default muxer (no `with_rf64`) writes no placeholder and is
    /// byte-identical to the historical output: plain `RIFF`, no `JUNK`
    /// chunk, no `ds64`.
    #[test]
    fn rf64_default_writes_no_placeholder() {
        let payload = vec![0u8; 2 * 64];
        let stream = make_stream(SampleFormat::S16, 1, 48_000);
        let plain = mux_to_bytes(&stream, &payload, WavMuxOptions::default(), "rf64-none-a");
        let explicit = mux_to_bytes(
            &stream,
            &payload,
            WavMuxOptions::default().with_rf64(Rf64Mode::Never),
            "rf64-none-b",
        );
        assert_eq!(plain, explicit);
        assert_eq!(&plain[0..4], b"RIFF");
        // The chunk right after WAVE is `fmt `, not a placeholder.
        assert_eq!(&plain[12..16], b"fmt ");
    }

    /// A seekable sink that materialises only the bytes written below
    /// `keep_below` (the header region) but still tracks a 64-bit
    /// high-water mark for everything beyond it. This lets a test drive
    /// the muxer through a genuine >4 GiB `data` payload — exercising the
    /// real `Rf64Mode::Reserve` overflow→promotion branch — without
    /// allocating gigabytes. Bytes written at or above `keep_below` are
    /// discarded; the header bytes (and the trailer's seek-back patches)
    /// are retained so the resulting prefix can be inspected.
    struct SparseSink {
        head: Vec<u8>,
        keep_below: u64,
        pos: u64,
        len: u64,
    }

    impl SparseSink {
        fn new(keep_below: u64) -> Self {
            SparseSink {
                head: Vec::new(),
                keep_below,
                pos: 0,
                len: 0,
            }
        }
    }

    impl std::io::Write for SparseSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let end = self.pos + buf.len() as u64;
            if self.pos < self.keep_below {
                // The (possibly partial) portion that lands in the kept
                // header region is materialised at its absolute offset.
                let kept = (self.keep_below - self.pos).min(buf.len() as u64) as usize;
                let start = self.pos as usize;
                if self.head.len() < start + kept {
                    self.head.resize(start + kept, 0);
                }
                self.head[start..start + kept].copy_from_slice(&buf[..kept]);
            }
            self.pos = end;
            self.len = self.len.max(end);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl std::io::Seek for SparseSink {
        fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
            self.pos = match from {
                SeekFrom::Start(n) => n,
                SeekFrom::End(d) => (self.len as i64 + d) as u64,
                SeekFrom::Current(d) => (self.pos as i64 + d) as u64,
            };
            Ok(self.pos)
        }
    }

    /// `Rf64Mode::Reserve` promotes its `JUNK` placeholder to a real
    /// `ds64` chunk and flips the magic to `RF64` once the finished
    /// payload overflows a 32-bit size field — the BS.2088-2 §4.2
    /// on-the-fly conversion. Driven through a sparse sink so the test
    /// can stream a >4 GiB `data` chunk without allocating it. The sink
    /// is shared by `Rc<RefCell<…>>` so the retained header bytes are
    /// inspectable after the boxed muxer drops its handle.
    #[test]
    fn rf64_reserve_overflow_promotes_to_rf64() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let stream = make_stream(SampleFormat::S16, 1, 48_000);
        let opts = WavMuxOptions::default().with_rf64(Rf64Mode::Reserve);

        // Keep the first 64 KiB (the header + placeholder live there).
        let sink = Rc::new(RefCell::new(SparseSink::new(64 * 1024)));
        // 4 GiB + a little, streamed in 256 MiB zero-PCM chunks.
        let chunk = vec![0u8; 256 * 1024 * 1024];
        let total: u64 = 4 * 1024 * 1024 * 1024 + 8 * 1024 * 1024;
        {
            let ws: Box<dyn WriteSeek> = Box::new(SharedSink(Rc::clone(&sink)));
            let mut mux = open_muxer_with(ws, std::slice::from_ref(&stream), opts).unwrap();
            mux.write_header().unwrap();
            let mut written: u64 = 0;
            while written < total {
                let n = chunk.len().min((total - written) as usize);
                mux.write_packet(&Packet::new(0, stream.time_base, chunk[..n].to_vec()))
                    .unwrap();
                written += n as u64;
            }
            mux.write_trailer().unwrap();
        }

        let head = sink.borrow().head.clone();
        // Magic promoted to RF64; legacy RIFF size = sentinel.
        assert_eq!(&head[0..4], b"RF64");
        assert_eq!(
            u32::from_le_bytes([head[4], head[5], head[6], head[7]]),
            SIZE64_SENTINEL
        );
        // The JUNK placeholder was promoted to a ds64 chunk in place.
        assert_eq!(&head[12..16], b"ds64");
        // `data` 32-bit size field also carries the sentinel.
        let body = &head[20..20 + DS64_FIXED_BODY_LEN as usize];
        let riff_size = u64::from_le_bytes(body[0..8].try_into().unwrap());
        let data_size = u64::from_le_bytes(body[8..16].try_into().unwrap());
        let sample_count = u64::from_le_bytes(body[16..24].try_into().unwrap());
        assert_eq!(data_size, total);
        // S16 mono → 1 frame == 2 bytes; per-channel sample count.
        assert_eq!(sample_count, total / 2);
        // riffSize = everything after the 8-byte RIFF+size header:
        // WAVE(4) + ds64(8+28) + fmt (8+16) + data header(8) + payload.
        let overhead = 4 + (8 + DS64_FIXED_BODY_LEN as u64) + (8 + 16) + 8;
        assert_eq!(riff_size, total + overhead);
    }

    // ---- cue / plst / LIST adtl: typed parse/to_bytes + muxer round-trip ----

    /// `CuePoint`/`CueChunk` parse and to_bytes are exact inverses, and
    /// `at_sample` fills the single-`data`-chunk convention fields.
    #[test]
    fn cue_chunk_to_bytes_roundtrip() {
        let cue = CueChunk::new(vec![
            CuePoint::at_sample(1, 0),
            CuePoint::at_sample(2, 44_100),
            CuePoint {
                name: 7,
                position: 96_000,
                fcc_chunk: *b"slnt",
                chunk_start: 12,
                block_start: 34,
                sample_offset: 56,
            },
        ]);
        let body = cue.to_bytes();
        assert_eq!(body.len(), cue.body_len());
        assert_eq!(body.len() % 2, 0); // always even
        assert_eq!(u32::from_le_bytes(body[0..4].try_into().unwrap()), 3);
        let reparsed = CueChunk::parse(&body).unwrap();
        assert_eq!(reparsed, cue);
        // Convention fields for a single-data file.
        assert_eq!(cue.points[1].fcc_chunk, *b"data");
        assert_eq!(cue.points[1].position, 44_100);
        assert_eq!(cue.points[1].sample_offset, 44_100);
        assert_eq!(cue.points[1].chunk_start, 0);
    }

    /// A `cue ` claiming more points than the body carries is clamped.
    #[test]
    fn cue_chunk_parse_clamps_overclaimed_count() {
        let mut body = Vec::new();
        body.extend_from_slice(&9u32.to_le_bytes()); // claims 9
        body.extend_from_slice(&CuePoint::at_sample(1, 100).to_bytes()); // only 1 fits
        let cue = CueChunk::parse(&body).unwrap();
        assert_eq!(cue.points.len(), 1);
        assert_eq!(cue.points[0].name, 1);
        // Body shorter than the 4-byte count header → None.
        assert!(CueChunk::parse(&[0u8; 3]).is_none());
    }

    /// `PlaylistSegment`/`PlaylistChunk` parse/to_bytes inverses.
    #[test]
    fn plst_chunk_to_bytes_roundtrip() {
        let plst = PlaylistChunk::new(vec![
            PlaylistSegment {
                cue_id: 1,
                length: 22_050,
                loops: 2,
            },
            PlaylistSegment {
                cue_id: 1, // same cue replayed
                length: 22_050,
                loops: 1,
            },
        ]);
        let body = plst.to_bytes();
        assert_eq!(body.len(), plst.body_len());
        assert_eq!(body.len() % 2, 0);
        assert_eq!(PlaylistChunk::parse(&body).unwrap(), plst);
    }

    /// `AdtlChunk` parse/to_list_body inverses across all three entry
    /// kinds, including odd-length `labl` text (word-alignment pad).
    #[test]
    fn adtl_chunk_to_list_body_roundtrip() {
        let adtl = AdtlChunk::new(vec![
            AdtlEntry::Label {
                name: 1,
                text: "intro".to_string(), // 5 chars + NUL = 6 bytes body → odd dwName? body=4+6=10 even
            },
            AdtlEntry::Note {
                name: 1,
                text: "comment text".to_string(),
            },
            AdtlEntry::LabeledText {
                name: 2,
                sample_length: 1024,
                purpose: *b"capt",
                country: 1,
                language: 9,
                dialect: 1,
                code_page: 1252,
                text: "caption".to_string(),
            },
        ]);
        let body = adtl.to_list_body();
        assert_eq!(body.len(), adtl.list_body_len());
        assert_eq!(body.len() % 2, 0); // list body always even
        assert_eq!(&body[0..4], b"adtl");
        let reparsed = AdtlChunk::parse(&body[4..]);
        assert_eq!(reparsed, adtl);
    }

    /// Full muxer → demuxer round-trip: a WAV written with cue / plst /
    /// adtl trailing chunks reads back with the typed accessors intact,
    /// the payload unchanged, and the chunks correctly placed *after*
    /// `data` (verified by the demuxer surfacing them despite the
    /// post-data position).
    #[test]
    fn mux_demux_cue_plst_adtl_roundtrip() {
        let samples: Vec<i16> = (0..500).map(|i| (i as i16).wrapping_mul(53)).collect();
        let mut payload = Vec::with_capacity(samples.len() * 2);
        for s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }
        let stream = make_stream(SampleFormat::S16, 1, 44_100);

        let cue = CueChunk::new(vec![CuePoint::at_sample(1, 0), CuePoint::at_sample(2, 250)]);
        let plst = PlaylistChunk::new(vec![PlaylistSegment {
            cue_id: 1,
            length: 250,
            loops: 3,
        }]);
        let adtl = AdtlChunk::new(vec![
            AdtlEntry::Label {
                name: 1,
                text: "start".to_string(),
            },
            AdtlEntry::Note {
                name: 2,
                text: "midpoint marker".to_string(),
            },
            AdtlEntry::LabeledText {
                name: 1,
                sample_length: 250,
                purpose: *b"rgn ",
                country: 0,
                language: 0,
                dialect: 0,
                code_page: 0,
                text: "first half".to_string(),
            },
        ]);

        let opts = WavMuxOptions::default()
            .with_cue(cue.clone())
            .with_plst(plst.clone())
            .with_adtl(adtl.clone());
        let bytes = mux_to_bytes(&stream, &payload, opts, "cue-plst-adtl");

        // The trailing chunks must sit after `data` — confirm `data`
        // appears before `cue ` / `plst` / `LIST` in the byte stream.
        let find = |needle: &[u8]| {
            bytes
                .windows(needle.len())
                .position(|w| w == needle)
                .unwrap()
        };
        let data_pos = find(b"data");
        assert!(find(b"cue ") > data_pos);
        assert!(find(b"plst") > data_pos);
        assert!(find(b"adtl") > data_pos);

        let typed = open_wav_demuxer(Box::new(std::io::Cursor::new(bytes.clone()))).unwrap();
        assert_eq!(typed.cue(), Some(&cue));
        assert_eq!(typed.plst(), Some(&plst));
        assert_eq!(typed.adtl(), Some(&adtl));

        // Payload still decodes intact through the dyn-Demuxer path.
        let mut dmx = open_demux_from_bytes(bytes);
        let mut out = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out, payload);
    }

    /// A WAV with no cue / plst / adtl reads back with all three typed
    /// accessors `None`, and the byte stream is unchanged from the
    /// pre-feature muxer (no stray trailing chunks).
    #[test]
    fn mux_without_cue_plst_adtl_is_unchanged() {
        let payload = vec![0u8; 64];
        let stream = make_stream(SampleFormat::U8, 1, 8_000);
        let plain = mux_to_bytes(&stream, &payload, WavMuxOptions::default(), "no-cue");
        let typed = open_wav_demuxer(Box::new(std::io::Cursor::new(plain.clone()))).unwrap();
        assert!(typed.cue().is_none());
        assert!(typed.plst().is_none());
        assert!(typed.adtl().is_none());
        // No trailing-chunk ids leaked into the output.
        assert!(!plain.windows(4).any(|w| w == b"cue "));
        assert!(!plain.windows(4).any(|w| w == b"plst"));
    }

    /// A seekable sink shared via `Rc<RefCell<…>>` so a test can read the
    /// retained header bytes after the boxed muxer drops its handle.
    /// Single-threaded (one test), so the `Send` bound on `WriteSeek` is
    /// satisfied by the wrapper's manual `unsafe impl Send`.
    struct SharedSink(std::rc::Rc<std::cell::RefCell<SparseSink>>);
    // Used only single-threaded inside one test.
    unsafe impl Send for SharedSink {}
    impl std::io::Write for SharedSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.borrow_mut().write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.0.borrow_mut().flush()
        }
    }
    impl std::io::Seek for SharedSink {
        fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
            self.0.borrow_mut().seek(from)
        }
    }
}
