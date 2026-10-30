"""Blizzard's TACT CDN and the client's local CASC storage: adding an older StarCraft II build
to an existing install so the engine can play that build's replays.

A replay only plays in the engine build it was recorded on (`Versions/Base<build>/SC2_x64.exe`,
launched with `-dataVersion <root key>` where the root key is the replay header's
`m_ngdpRootKey`). The Battle.net client downloads such a build on demand when a player opens an
old replay; this module does the same without the client, from the CDN (requests only):

  BuildConfig (hash from BlizzTrack / README table)
    -> encoding file         CKey (MD5 of content) -> EKey (MD5 of the BLTE-encoded file)
    -> install manifest      loose files to write to disk (Versions/Base<build>/SC2_x64.exe ...)
    -> download manifest     every EKey the build needs, with tags (Windows/OSX, locale, ...)
  CDNConfig -> archive indexes: where each EKey lives (archive + offset + size); the rest are
    loose at tpr/sc2/data/xx/yy/<ekey>

and writes what the install does not have yet into its CASC storage (`SC2Data/data`), exactly
as the Battle.net Agent does: BLTE files appended to the 1 GB `data.NNN` archives behind a
30-byte header, one `.idx` file per key bucket (new version per bucket), the `shmem` free-space
table, the BuildConfig copied to `SC2Data/config/xx/yy/`, and a row in `.build.info` (the
engine reads the rows, loads each BuildConfig and picks the one whose root matches
-dataVersion). Formats: https://wowdev.wiki/CASC and https://wowdev.wiki/TACT; the checksum
and index-hash algorithms below were verified against the existing storage before use.

Usage (from the repo root, with the Wine install found by wine_run_config):

    .venv/bin/python -m cosmicstar.envs.tact 93333 --dry-run    # just report what is missing
    .venv/bin/python -m cosmicstar.envs.tact 93333 --force      # install (--force: while engines run)
    .venv/bin/python -m cosmicstar.envs.tact 93333 --install "<other install>" --force
    .venv/bin/python -m cosmicstar.envs.tact 93333 --revert "<install>/SC2Data/backup-<time>"

Every install backs up .build.info, shmem and the live .idx files first; --revert undoes it
exactly. Installing into a copy of the install (archives hardlinked, the rest copied; README
"Older builds from the CDN") leaves the real one untouched; scripts then use it via SC2PATH.

KNOWN_BUILDS maps base builds to BuildConfig hashes; any other build can be given as
`<build>=<buildconfig hash>` (hashes: https://blizztrack.com/api/manifest/s2/versions?seqn=...).
"""
import collections
import glob
import hashlib
import io
import os
import shutil
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

CDN_HOST = "http://level3.blizzard.com"
CDN_PATH = "tpr/sc2"
# base build -> BuildConfig hash (worldmodel/README.md, "Older builds from the CDN")
KNOWN_BUILDS = {
    93333: "8453c2f1c98b955334c7284215429c36",  # 5.0.14
    94137: "05cce99544c8b80c07f6eb3250b957a9",  # 5.0.14
    95841: "f48701eb2ef86e687658d3deb0fc1bc8",  # 5.0.15
    97364: "2e257e413468e65da51630c5080a6168",  # 5.0.16
    97563: "5bc8dcc1fade3a320c12936586b7c0ed",  # 5.0.16
    98310: "1d231af579faf03e61a364722a3b87b4",  # 5.0.17 (the installed one)
}
DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "../../../data/tact")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------------------------
# CDN
# --------------------------------------------------------------------------------------------
class CDN:
    """Fetches from the TACT CDN with a local cache of configs and manifests (not archive
    ranges: those go straight into the storage)."""

    def __init__(self, cache_dir=DEFAULT_CACHE, host=CDN_HOST, path=CDN_PATH, threads=8):
        import requests
        self.requests = requests
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.host, self.path, self.threads = host, path, threads
        self._local = __import__("threading").local()

    def session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = self.requests.Session()
            a = self.requests.adapters.HTTPAdapter(max_retries=self.requests.adapters.Retry(
                total=6, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504)))
            s.mount("http://", a)
            s.mount("https://", a)
        return s

    def url(self, kind, key, suffix=""):
        return f"{self.host}/{self.path}/{kind}/{key[:2]}/{key[2:4]}/{key}{suffix}"

    def get(self, kind, key, suffix="", rng=None, timeout=300):
        headers = {"Range": f"bytes={rng[0]}-{rng[0] + rng[1] - 1}"} if rng else {}
        r = self.session().get(self.url(kind, key, suffix), headers=headers, timeout=timeout)
        r.raise_for_status()
        if rng and len(r.content) != rng[1]:
            raise IOError(f"{self.url(kind, key, suffix)} range {rng}: got {len(r.content)} bytes")
        return r.content

    def cached(self, kind, key, suffix=""):
        """config / data / index files by key, kept in the cache dir."""
        p = os.path.join(self.cache_dir, kind, key + suffix)
        if not os.path.exists(p):
            data = self.get(kind, key, suffix)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
        with open(p, "rb") as f:
            return f.read()


def parse_config(text):
    """`key = v1 v2` lines -> {key: [v1, v2]}."""
    out = {}
    for line in text.decode().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.split()
    return out


# --------------------------------------------------------------------------------------------
# BLTE
# --------------------------------------------------------------------------------------------
def blte_decode(data):
    """Decodes a BLTE-encoded file (zlib 'Z' and raw 'N' chunks; 'F' frames recursively)."""
    if data[:4] != b"BLTE":
        raise ValueError("not BLTE")
    hsize = struct.unpack_from(">I", data, 4)[0]
    if hsize == 0:
        return _blte_chunk(data[8:])
    count = struct.unpack_from(">I", data, 8)[0] & 0xFFFFFF
    chunks, pos = [], 12
    for _ in range(count):
        csize, dsize = struct.unpack_from(">II", data, pos)
        chunks.append((csize, dsize, data[pos + 8:pos + 24]))
        pos += 24
    out, pos = [], hsize
    for csize, dsize, md5 in chunks:
        chunk = data[pos:pos + csize]
        if hashlib.md5(chunk).digest() != md5:
            raise ValueError("BLTE chunk checksum mismatch")
        dec = _blte_chunk(chunk)
        if len(dec) != dsize:
            raise ValueError(f"BLTE chunk size {len(dec)} != {dsize}")
        out.append(dec)
        pos += csize
    return b"".join(out)


def _blte_chunk(chunk):
    mode = chunk[0:1]
    if mode == b"N":
        return chunk[1:]
    if mode == b"Z":
        return zlib.decompress(chunk[1:])
    if mode == b"F":
        return blte_decode(chunk[1:])
    raise ValueError(f"unsupported BLTE chunk mode {mode!r} (encrypted?)")


def blte_ekey(data):
    """The EKey of a BLTE file: MD5 of the header (through the chunk table) when there is one,
    else of the whole file."""
    hsize = struct.unpack_from(">I", data, 4)[0]
    return hashlib.md5(data[:hsize] if hsize else data).digest()


# --------------------------------------------------------------------------------------------
# TACT manifests
# --------------------------------------------------------------------------------------------
def _cstr(buf, pos):
    end = buf.index(b"\0", pos)
    return buf[pos:end].decode("utf-8"), end + 1


def parse_encoding(data):
    """CKey -> [EKey, ...] from a decoded encoding file (CKey pages only)."""
    if data[:2] != b"EN":
        raise ValueError("not an encoding file")
    ck_size, ek_size = data[3], data[4]
    ck_page_kb, ek_page_kb, ck_pages, ek_pages = struct.unpack_from(">HHII", data, 5)
    espec_size = struct.unpack_from(">I", data, 18)[0]
    pos = 22 + espec_size
    pos += ck_pages * (ck_size + 16)  # page index
    page_size = ck_page_kb * 1024
    out = {}
    for _ in range(ck_pages):
        page, p = data[pos:pos + page_size], 0
        while p + 6 + ck_size <= len(page):
            n = page[p]
            if n == 0:
                break
            ckey = page[p + 6:p + 6 + ck_size]
            p += 6 + ck_size
            out[ckey] = [page[p + i * ek_size:p + (i + 1) * ek_size] for i in range(n)]
            p += n * ek_size
        pos += page_size
    return out


def _tags(data, pos, num_tags, num_entries):
    mask_len = (num_entries + 7) // 8
    tags = {}
    for _ in range(num_tags):
        name, pos = _cstr(data, pos)
        typ = struct.unpack_from(">H", data, pos)[0]
        mask = data[pos + 2:pos + 2 + mask_len]
        tags[name] = (typ, mask)
        pos += 2 + mask_len
    return tags, pos


def _has(mask, i):
    return (mask[i >> 3] >> (7 - (i & 7))) & 1


def parse_install(data):
    """(entries [(name, CKey, size)], tags {name: (type, mask)})."""
    if data[:2] != b"IN":
        raise ValueError("not an install manifest")
    hash_size = data[3]
    num_tags, num_entries = struct.unpack_from(">HI", data, 4)
    tags, pos = _tags(data, 10, num_tags, num_entries)
    entries = []
    for _ in range(num_entries):
        name, pos = _cstr(data, pos)
        ckey = data[pos:pos + hash_size]
        size = struct.unpack_from(">I", data, pos + hash_size)[0]
        entries.append((name, ckey, size))
        pos += hash_size + 4
    return entries, tags


def parse_download(data):
    """(entries [(EKey, size, priority)], tags {name: (type, mask)})."""
    if data[:2] != b"DL":
        raise ValueError("not a download manifest")
    version, ek_size, has_checksum = data[2], data[3], data[4]
    num_entries, num_tags = struct.unpack_from(">IH", data, 5)
    pos = 11
    flag_bytes = 0
    if version >= 2:
        flag_bytes = data[pos]
        pos += 1
    if version >= 3:
        pos += 4
    entries = []
    esize = ek_size + 5 + 1 + (4 if has_checksum else 0) + flag_bytes
    for _ in range(num_entries):
        ekey = data[pos:pos + ek_size]
        size = int.from_bytes(data[pos + ek_size:pos + ek_size + 5], "big")
        prio = data[pos + ek_size + 5]
        entries.append((ekey, size, prio))
        pos += esize
    tags, pos = _tags(data, pos, num_tags, num_entries)
    return entries, tags


CODE_TAG_TYPE = 16384  # "code": the binaries, which the Agent writes to disk (install manifest) and
                       # never keeps in the storage


def select_by_tags(tags, num_entries, tag_configs):
    """Indexes of the download-manifest entries an install with the given .build.info tag
    strings keeps in its storage. Each config ("Windows code US? acct-USA? geoip-US? enUS
    speech?") names tags (trailing '?' = optional; names that are not tags are ignored); an
    entry matches a config when, for every tag type the config names (platform, locale,
    region, speech/text), it carries one of the named tags; the result is the union over the
    configs, minus the "code" entries (checked against the installed 5.0.17: the rule selects
    188 576 of its manifest's 608 424 entries, 188 448 of which the storage has, and 192 of the
    entries it rejects are present too, i.e. it reproduces what the Agent installed to 0.1%)."""
    by_type = collections.defaultdict(list)
    for name, (typ, mask) in tags.items():
        by_type[typ].append((name, mask))
    code_masks = [m for _, m in by_type.get(CODE_TAG_TYPE, [])]
    chosen = set()
    for cfg in tag_configs:
        names = {t.rstrip("?") for t in cfg.split()}
        groups = []
        for typ, lst in by_type.items():
            if typ == CODE_TAG_TYPE:
                continue
            sel = [mask for name, mask in lst if name in names]
            if sel:
                groups.append(sel)
        for i in range(num_entries):
            if any(_has(m, i) for m in code_masks):
                continue
            if all(any(_has(m, i) for m in sel) for sel in groups):
                chosen.add(i)
    return chosen


def parse_archive_index(data):
    """{EKey: (offset, size)} of one CDN archive's .index file."""
    footer = data[-28:]
    key_size, num = footer[14], struct.unpack_from("<I", footer, 16)[0]
    block = 4096
    nblocks = (len(data) - 28) // (block + key_size + 8)
    out = {}
    for b in range(nblocks):
        base = b * block
        for i in range(block // (key_size + 8)):
            p = base + i * (key_size + 8)
            ekey = data[p:p + key_size]
            if not any(ekey):
                break
            size, offset = struct.unpack_from(">II", data, p + key_size)
            out[ekey] = (offset, size)
            if len(out) == num:
                return out
    return out


# --------------------------------------------------------------------------------------------
# Jenkins lookup3 (the checksums the local storage uses)
# --------------------------------------------------------------------------------------------
_M = 0xFFFFFFFF


def _rot(x, k):
    return ((x << k) | (x >> (32 - k))) & _M


def hashlittle2(data, pc=0, pb=0):
    """Bob Jenkins' lookup3 hashlittle2: returns (pc, pb)."""
    length = len(data)
    a = b = c = (0xdeadbeef + length + pc) & _M
    c = (c + pb) & _M
    i = 0
    while length > 12:
        a = (a + struct.unpack_from("<I", data, i)[0]) & _M
        b = (b + struct.unpack_from("<I", data, i + 4)[0]) & _M
        c = (c + struct.unpack_from("<I", data, i + 8)[0]) & _M
        a = (a - c) & _M; a ^= _rot(c, 4); c = (c + b) & _M
        b = (b - a) & _M; b ^= _rot(a, 6); a = (a + c) & _M
        c = (c - b) & _M; c ^= _rot(b, 8); b = (b + a) & _M
        a = (a - c) & _M; a ^= _rot(c, 16); c = (c + b) & _M
        b = (b - a) & _M; b ^= _rot(a, 19); a = (a + c) & _M
        c = (c - b) & _M; c ^= _rot(b, 4); b = (b + a) & _M
        length -= 12
        i += 12
    if length == 0:
        return c, b
    tail = data[i:] + b"\0" * (12 - length)
    a = (a + struct.unpack_from("<I", tail, 0)[0]) & _M
    b = (b + struct.unpack_from("<I", tail, 4)[0]) & _M
    c = (c + struct.unpack_from("<I", tail, 8)[0]) & _M
    c ^= b; c = (c - _rot(b, 14)) & _M
    a ^= c; a = (a - _rot(c, 11)) & _M
    b ^= a; b = (b - _rot(a, 25)) & _M
    c ^= b; c = (c - _rot(b, 16)) & _M
    a ^= c; a = (a - _rot(c, 4)) & _M
    b ^= a; b = (b - _rot(a, 14)) & _M
    c ^= b; c = (c - _rot(b, 24)) & _M
    return c, b


_CHECKSUM_TABLE = [0x049396b8, 0x72a82a9b, 0xee626cca, 0x9917754f, 0x15de40b1, 0xf5a8a9b6, 0x421eac7e, 0xa9d55c9a,
                   0x317fd40c, 0x04faf80d, 0x3d6be971, 0x52933cfd, 0x27f64b7d, 0xc6f5c11b, 0xd5757e3a, 0x6c388745]


def record_header(ekey, size, archive, offset, flags=0):
    """The 30-byte header in front of a file in data.NNN: reversed EKey, total size (header
    included), flags, checksum A (lookup3 of the first 0x16 bytes), checksum B (of the first
    0x1A bytes, keyed by the position in the archive)."""
    h = bytearray(ekey[::-1]) + struct.pack("<IH", size, flags)
    h += struct.pack("<I", hashlittle2(bytes(h), 0x3D6BE971)[0])
    off = ((offset & 0x3FFFFFFF) | ((archive & 3) << 30)) & _M
    enc = (_CHECKSUM_TABLE[(off + 0x1E) & 0xF] ^ ((off + 0x1E) & _M)).to_bytes(4, "little")
    hh = bytearray(4)
    for i in range(0x1A):
        hh[(i + off) & 3] ^= h[i]
    h += bytes(hh[(j + 0x1A + off) & 3] ^ enc[(j + 0x1A + off) & 3] for j in range(4))
    return bytes(h)


# --------------------------------------------------------------------------------------------
# Local CASC storage
# --------------------------------------------------------------------------------------------
IDX_HEADER = 0x28
IDX_ENTRY = 18
ARCHIVE_MAX = 0x40000000  # 30-bit offsets: 1 GB per data.NNN
MARKER_TAIL = bytes.fromhex("0038bfe1989077800be3361b4e20")  # the Agent's per-archive marker keys
FREE_ENTRIES = 1090


def bucket_of(key):
    i = 0
    for b in key[:9]:
        i ^= b
    return (i & 0xF) ^ (i >> 4)


class LocalStorage:
    """SC2Data/data: the data.NNN archives, their .idx files and shmem."""

    def __init__(self, data_dir):
        self.dir = data_dir
        self.versions = {}  # bucket -> newest idx version
        for f in glob.glob(os.path.join(data_dir, "*.idx")):
            n = os.path.basename(f)
            b, v = int(n[:2], 16), int(n[2:10], 16)
            if v > self.versions.get(b, -1):
                self.versions[b] = v
        self.entries = {}  # 9-byte key -> (archive, offset, size)
        for b, v in self.versions.items():
            self.entries.update(self.read_idx(os.path.join(data_dir, f"{b:02x}{v:08x}.idx")))
        self.read_shmem()

    @staticmethod
    def read_idx(path):
        with open(path, "rb") as f:
            data = f.read()
        hsize, hhash, version = struct.unpack_from("<IIH", data, 0)
        if version != 7 or hashlittle2(data[8:8 + hsize])[0] != hhash:
            raise ValueError(f"{path}: unexpected header")
        esize = struct.unpack_from("<I", data, 0x20)[0]
        out = {}
        for p in range(IDX_HEADER, IDX_HEADER + esize, IDX_ENTRY):
            raw = int.from_bytes(data[p + 9:p + 14], "big")
            out[data[p:p + 9]] = (raw >> 30, raw & 0x3FFFFFFF, struct.unpack_from("<I", data, p + 14)[0])
        return out

    def has(self, ekey):
        return ekey[:9] in self.entries

    def read_shmem(self):
        with open(os.path.join(self.dir, "shmem"), "rb") as f:
            s = f.read()
        btype, nxt = struct.unpack_from("<II", s, 0)
        if btype != 4:
            raise ValueError(f"shmem: unexpected block type {btype}")
        self.shmem_head = s[:nxt]
        nrec = (nxt - 0x108 - 64) // 8
        vers = struct.unpack_from("<16I", s, 0x108 + 8 * nrec)
        if any(vers[b] != self.versions.get(b) for b in range(16)):
            raise ValueError(f"shmem idx versions {vers} differ from the .idx files {self.versions}")
        ftype, count = struct.unpack_from("<II", s, nxt)
        if ftype != 1:
            raise ValueError(f"shmem: unexpected free-space block type {ftype}")
        base = nxt + 0x20
        self.free = []
        for i in range(count):
            size = int.from_bytes(s[base + 5 * i:base + 5 * i + 5], "big")
            raw = int.from_bytes(s[base + 5 * FREE_ENTRIES + 5 * i:base + 5 * FREE_ENTRIES + 5 * i + 5], "big")
            self.free.append([raw >> 30, raw & 0x3FFFFFFF, size])  # archive, offset, free bytes
        self.shmem_len = len(s)

    def archives(self):
        return sorted(int(f[-3:]) for f in glob.glob(os.path.join(self.dir, "data.[0-9][0-9][0-9]")))

    # -- writing ------------------------------------------------------------------------------
    def allocate(self, size):
        """(archive, offset) of `size` free bytes, taken from the free list (lowest archive
        first, as the Agent fills them)."""
        for e in sorted(self.free, key=lambda e: (e[0], e[1])):
            if e[2] >= size:
                arch, off = e[0], e[1]
                e[1] += size
                e[2] -= size
                return arch, off
        raise IOError("no free space in the storage's free list")

    def write_record(self, ekey, blob, flags=0):
        """Appends one BLTE file to an archive; returns the index entry."""
        size = 0x1E + len(blob)
        if size > ARCHIVE_MAX:
            raise ValueError("file larger than an archive")
        arch, off = self.allocate(size)
        path = os.path.join(self.dir, f"data.{arch:03d}")
        if not os.path.exists(path):
            self.new_archive(arch, path)
        with open(path, "r+b") as f:
            f.seek(off)
            f.write(record_header(ekey, size, arch, off, flags))
            f.write(blob)
        self.pending[ekey[:9]] = (arch, off, size)
        return arch, off, size

    def new_archive(self, arch, path):
        """A fresh data.NNN starts with the Agent's 16 marker records (one per bucket)."""
        with open(path, "wb") as f:
            pass
        keys = [bytes([b0, arch]) + MARKER_TAIL for b0 in range(16)]
        keys.sort(key=bucket_of)
        for i, k in enumerate(keys):
            off = i * 0x1E
            with open(path, "r+b") as f:
                f.seek(off)
                f.write(record_header(k, 0x1E, arch, off, flags=1))
            self.pending[k[:9]] = (arch, off, 0x1E)
        for e in self.free:
            if e[0] == arch and e[1] == 0:
                e[1] += 16 * 0x1E
                e[2] -= 16 * 0x1E

    def begin(self):
        self.pending = {}

    def commit(self):
        """Writes a new .idx version for every bucket with new entries, then shmem."""
        by_bucket = collections.defaultdict(dict)
        for k, v in self.pending.items():
            by_bucket[bucket_of(k)][k] = v
        for b, new in by_bucket.items():
            old = self.read_idx(os.path.join(self.dir, f"{b:02x}{self.versions[b]:08x}.idx"))
            old.update(new)
            self.versions[b] += 1
            self.write_idx(os.path.join(self.dir, f"{b:02x}{self.versions[b]:08x}.idx"), b, old)
        self.entries.update(self.pending)
        self.pending = {}
        self.write_shmem()

    @staticmethod
    def write_idx(path, bucket, entries):
        body = b"".join(k + (((v[0] << 30) | v[1]).to_bytes(5, "big")) + struct.pack("<I", v[2])
                        for k, v in sorted(entries.items()))
        hdr = struct.pack("<HBBBBBBQ", 7, bucket, 0, 4, 5, 9, 30, 0x4000000000) + b"\0" * 8
        hdr = struct.pack("<II", 0x10, hashlittle2(hdr[:0x10])[0]) + hdr
        pc = pb = 0
        for p in range(0, len(body), IDX_ENTRY):
            pc, pb = hashlittle2(body[p:p + IDX_ENTRY], pc, pb)
        data = hdr + struct.pack("<II", len(body), pc) + body
        data += b"\0" * (-len(data) % 0x10000)
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

    def write_shmem(self):
        head = bytearray(self.shmem_head)
        nxt = struct.unpack_from("<I", head, 4)[0]
        nrec = (nxt - 0x108 - 64) // 8
        struct.pack_into("<16I", head, 0x108 + 8 * nrec, *[self.versions[b] for b in range(16)])
        free = [e for e in self.free if e[2] > 0]
        blk = struct.pack("<II", 1, len(free)) + b"\0" * 0x18
        sizes = b"".join(e[2].to_bytes(5, "big") for e in free)
        offs = b"".join(((e[0] << 30) | e[1]).to_bytes(5, "big") for e in free)
        blk += sizes + b"\0" * (5 * FREE_ENTRIES - len(sizes)) + offs + b"\0" * (5 * FREE_ENTRIES - len(offs))
        data = bytes(head) + blk
        data += b"\0" * max(0, self.shmem_len - len(data))
        path = os.path.join(self.dir, "shmem")
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
        self.free = free


# --------------------------------------------------------------------------------------------
# A build
# --------------------------------------------------------------------------------------------
class Build:
    """One build's TACT metadata, fetched lazily."""

    def __init__(self, cdn, build_config_hash):
        self.cdn = cdn
        self.hash = build_config_hash
        self.config = parse_config(cdn.cached("config", build_config_hash))
        self.root_ckey = self.config["root"][0]
        self.encoding_ekey = self.config["encoding"][1]
        self._encoding = None

    @property
    def encoding(self):
        if self._encoding is None:
            log(f"parsing encoding {self.encoding_ekey} ...")
            self._encoding = parse_encoding(blte_decode(self.cdn.cached("data", self.encoding_ekey)))
        return self._encoding

    def ekey(self, ckey_hex):
        return self.encoding[bytes.fromhex(ckey_hex)][0]

    def manifest(self, name):
        """Decoded install / download / size file (their EKeys are in the BuildConfig)."""
        return blte_decode(self.cdn.cached("data", self.config[name][1]))

    def metadata_ekeys(self):
        """EKeys of the build's own metadata files, which the storage keeps too."""
        keys = [self.encoding_ekey, self.config["install"][1], self.config["download"][1], self.config["size"][1]]
        keys.append(self.ekey(self.root_ckey).hex())
        return keys


class CDNIndex:
    """EKey -> (archive hash, offset, size) over all archives of a CDNConfig."""

    def __init__(self, cdn, cdn_config_hash):
        self.cdn = cdn
        cfg = parse_config(cdn.cached("config", cdn_config_hash))
        self.archives = cfg["archives"]
        self.where = {}
        log(f"loading {len(self.archives)} archive indexes ...")
        with ThreadPoolExecutor(cdn.threads) as ex:
            for arch, idx in zip(self.archives, ex.map(lambda a: cdn.cached("data", a, ".index"), self.archives)):
                for ekey, (off, size) in parse_archive_index(idx).items():
                    self.where[ekey] = (arch, off, size)

    def fetch(self, ekey):
        """The BLTE file for an EKey: from its archive, else loose."""
        w = self.where.get(ekey)
        if w:
            data = self.cdn.get("data", w[0], rng=(w[1], w[2]))
        else:
            data = self.cdn.get("data", ekey.hex())
        if data[:4] != b"BLTE":
            raise IOError(f"{ekey.hex()}: not a BLTE file")
        if blte_ekey(data) != ekey:
            raise IOError(f"{ekey.hex()}: EKey mismatch")
        return data


# --------------------------------------------------------------------------------------------
# The install
# --------------------------------------------------------------------------------------------
BUILD_INFO_HEADER = ("Branch!STRING:0|Active!DEC:1|Build Key!HEX:16|CDN Key!HEX:16|Install Key!HEX:16|IM Size!DEC:4|"
                     "CDN Path!STRING:0|CDN Hosts!STRING:0|CDN Servers!STRING:0|Tags!STRING:0|Armadillo!STRING:0|"
                     "Last Activated!STRING:0|Version!STRING:0|KeyRing!HEX:16|Product!STRING:0")


def read_build_info(install):
    path = os.path.join(install, ".build.info")
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    cols = [c.split("!")[0] for c in lines[0].split("|")]
    rows = [dict(zip(cols, l.split("|"))) for l in lines[1:]]
    return lines[0], rows


def plan(install, build, cdn_config_hash=None, cache=DEFAULT_CACHE, threads=8):
    """What installing `build` (base build number) into `install` takes. Returns a dict with
    the Build, the CDNIndex, the missing EKeys with sizes, and the install files to write."""
    bc_hash = KNOWN_BUILDS.get(build) if isinstance(build, int) else None
    if isinstance(build, str) and "=" in build:
        build, bc_hash = build.split("=")
        build = int(build)
    if not bc_hash:
        raise ValueError(f"no BuildConfig hash known for build {build}; pass <build>=<hash>")
    header, rows = read_build_info(install)
    active = [r for r in rows if r["Active"] == "1"] or rows
    tag_configs = active[0]["Tags"].split(":")
    cdn_config_hash = cdn_config_hash or active[0]["CDN Key"]
    cdn = CDN(cache, threads=threads)
    b = Build(cdn, bc_hash)
    storage = LocalStorage(os.path.join(install, "SC2Data", "data"))
    log(f"build {build}: BuildConfig {bc_hash}, root {b.root_ckey}; storage has {len(storage.entries)} entries "
        f"in {len(storage.archives())} archives")

    entries, tags = parse_download(b.manifest("download"))
    chosen = select_by_tags(tags, len(entries), tag_configs)
    need = {entries[i][0]: entries[i][1] for i in chosen}
    for k in b.metadata_ekeys():
        need.setdefault(bytes.fromhex(k), 0)
    missing = {k: s for k, s in need.items() if not storage.has(k)}
    log(f"download manifest: {len(entries)} files, {len(chosen)} for tags {tag_configs}; "
        f"missing locally: {len(missing)} ({sum(missing.values()) / 1e6:.1f} MB)")

    inst, itags = parse_install(b.manifest("install"))
    want_tags = {"Windows", "x86_64", "enUS"} & set(itags)
    files = []
    for i, (name, ckey, size) in enumerate(inst):
        if not all(_has(itags[t][1], i) for t in want_tags if itags[t][1]):
            continue
        dest = os.path.join(install, name.replace("\\", "/"))
        if name.startswith(f"Versions/Base{build}/") or not os.path.exists(dest):
            files.append((name, ckey, size))
    log(f"install manifest: {len(inst)} files; to write: {[f[0] for f in files]}")
    return dict(build=build, bc_hash=bc_hash, cdn=cdn, b=b, storage=storage, missing=missing, files=files,
                cdn_config_hash=cdn_config_hash, header=header, rows=rows, active=active[0])


def sc2_running(max_age=None):
    """Running processes that use the storage (the engine, the Agent), as (age in seconds,
    name); with max_age only the ones younger than that."""
    import subprocess
    out = subprocess.run(["ps", "-eo", "etimes,args"], capture_output=True, text=True).stdout
    names = ("sc2_x64.exe", "sc2.exe", "agent.exe", "sc2switcher_x64.exe", "sc2editor_x64.exe")
    found = []
    for line in out.splitlines()[1:]:
        age, _, args = line.strip().partition(" ")
        low = args.lower()
        for n in names:
            if n in low and (max_age is None or int(age) < max_age):
                found.append((int(age), n))
                break
    return sorted(found)


def fetch_missing(p, index):
    """Downloads the build's missing BLTE files into the cache ("blobs/<ekey>"), verified
    against their EKey; returns {ekey: path}. No storage writes, so this can run while the
    engine is playing."""
    cdn = p["cdn"]
    blob_dir = os.path.join(cdn.cache_dir, "blobs")
    os.makedirs(blob_dir, exist_ok=True)
    missing = sorted(p["missing"], key=lambda k: (index.where.get(k, ("~",))[0], index.where.get(k, (0, 0))[1]))
    paths = {k: os.path.join(blob_dir, k.hex()) for k in missing}
    todo = [k for k in missing if not os.path.exists(paths[k])]
    done = [0, 0]

    def fetch(ekey):
        data = index.fetch(ekey)
        with open(paths[ekey] + ".part", "wb") as f:
            f.write(data)
        os.replace(paths[ekey] + ".part", paths[ekey])
        return len(data)

    t0 = time.time()
    with ThreadPoolExecutor(cdn.threads) as ex:
        for n in ex.map(fetch, todo):
            done[0] += 1
            done[1] += n
            if done[0] % 50 == 0:
                log(f"  {done[0]}/{len(todo)} files, {done[1] / 1e6:.0f} MB")
    log(f"fetched {done[0]} files, {done[1] / 1e6:.1f} MB in {time.time() - t0:.0f} s "
        f"({len(missing) - len(todo)} were cached)")
    return paths


def write_storage(p, paths, force=False):
    """The critical section: appends the fetched files to the archives and rewrites the .idx
    files and shmem. The engine rewrites shmem when it starts (and nothing else in the
    storage), so this refuses while the engine or the Agent runs unless force; with force it
    waits for a moment when no engine instance is less than 20 s old, writes (a few seconds),
    and then checks shmem came through."""
    storage = p["storage"]
    running = sc2_running()
    if running and not force:
        raise RuntimeError(f"{running} running: the storage can only be written while nothing uses it "
                           "(--force writes between engine launches)")
    if running:
        log(f"{len(running)} engine/Agent processes running; waiting for a gap between launches")
        while sc2_running(max_age=20):
            time.sleep(1)
    t0 = time.time()
    storage.begin()
    n = size = 0
    for ekey, path in paths.items():
        with open(path, "rb") as f:
            blob = f.read()
        if blte_ekey(blob) != ekey:
            raise IOError(f"{path}: EKey mismatch")
        storage.write_record(ekey, blob)
        n += 1
        size += len(blob)
    storage.commit()
    log(f"storage: {n} files, {size / 1e6:.1f} MB written in {time.time() - t0:.1f} s; "
        f"archives now {storage.archives()[-1] + 1}, idx versions {[storage.versions[b] for b in range(16)]}")
    if running:
        time.sleep(25)  # an engine started meanwhile writes shmem within its first seconds
        check = LocalStorage(storage.dir)  # raises when shmem and the .idx files disagree
        if check.free != storage.free:
            log("shmem was rewritten by an engine start; writing it again")
            storage.write_shmem()


def backup(install, storage):
    """Copies what the install step rewrites (.build.info, shmem, the live .idx files) to
    SC2Data/backup-<time>/ with a manifest of the archive sizes, so revert() can undo it."""
    import json
    bak = os.path.join(install, "SC2Data", f"backup-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(bak, exist_ok=True)
    shutil.copy2(os.path.join(install, ".build.info"), bak)
    shutil.copy2(os.path.join(storage.dir, "shmem"), bak)
    for bk, v in storage.versions.items():
        shutil.copy2(os.path.join(storage.dir, f"{bk:02x}{v:08x}.idx"), bak)
    sizes = {a: os.path.getsize(os.path.join(storage.dir, f"data.{a:03d}")) for a in storage.archives()}
    json.dump(dict(archive_sizes=sizes, idx_versions=storage.versions), open(os.path.join(bak, "manifest.json"), "w"), indent=1)
    log(f"backed up .build.info, shmem and the .idx files to {bak}")
    return bak


def revert(install, bak, build=None):
    """Undoes an install_build() from its backup folder: restores .build.info, shmem and the
    .idx files, deletes newer .idx versions, truncates the archives to their recorded sizes
    and deletes archives created since; with build, also removes Versions/Base<build> and the
    build's BuildConfig copy. Nothing older than the backup is touched."""
    import json
    m = json.load(open(os.path.join(bak, "manifest.json")))
    data = os.path.join(install, "SC2Data", "data")
    sizes = {int(k): v for k, v in m["archive_sizes"].items()}
    for a in sorted(int(f[-3:]) for f in glob.glob(os.path.join(data, "data.[0-9][0-9][0-9]"))):
        p = os.path.join(data, f"data.{a:03d}")
        if a not in sizes:
            os.remove(p)
            log(f"removed {p}")
        elif os.path.getsize(p) != sizes[a]:
            with open(p, "r+b") as f:
                f.truncate(sizes[a])
            log(f"truncated data.{a:03d} to {sizes[a]}")
    versions = {int(k): v for k, v in m["idx_versions"].items()}
    for f in glob.glob(os.path.join(data, "*.idx")):
        n = os.path.basename(f)
        if int(n[2:10], 16) > versions[int(n[:2], 16)]:
            os.remove(f)
    for f in glob.glob(os.path.join(bak, "*.idx")) + [os.path.join(bak, "shmem")]:
        shutil.copy2(f, data)
    shutil.copy2(os.path.join(bak, ".build.info"), install)
    if build:
        shutil.rmtree(os.path.join(install, "Versions", f"Base{build}"), ignore_errors=True)
        h = KNOWN_BUILDS.get(build)
        if h:
            cfg = os.path.join(install, "SC2Data", "config", h[:2], h[2:4], h)
            if os.path.exists(cfg):
                os.remove(cfg)
    log(f"reverted to {bak}")


def install_build(install, build, dry_run=False, cache=DEFAULT_CACHE, threads=8, force=False):
    t0 = time.time()
    p = plan(install, build, cache=cache, threads=threads)
    b, storage, cdn = p["b"], p["storage"], p["cdn"]
    row_present = any(r["Build Key"] == p["bc_hash"] for r in p["rows"])
    if row_present and not p["missing"] and not p["files"]:
        log("already installed")
        return p
    if dry_run:
        return p
    index = CDNIndex(cdn, p["cdn_config_hash"])
    paths = fetch_missing(p, index)
    # loose files from the install manifest, decoded and verified in the cache first
    loose = []
    for name, ckey, size in p["files"]:
        ekey = b.encoding[ckey][0]
        blob = open(paths[ekey], "rb").read() if ekey in paths else (
            storage_read(storage, ekey) if storage.has(ekey) else index.fetch(ekey))
        content = blte_decode(blob)
        if hashlib.md5(content).digest() != ckey or len(content) != size:
            raise IOError(f"{name}: content hash mismatch")
        loose.append((name, content))

    backup(install, storage)
    if paths:
        write_storage(p, paths, force=force)
    for name, content in loose:
        dest = os.path.join(install, name.replace("\\", "/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".part", "wb") as f:
            f.write(content)
        os.replace(dest + ".part", dest)
        os.chmod(dest, 0o775 if dest.endswith(".exe") else 0o664)
        log(f"wrote {name} ({len(content) / 1e6:.1f} MB)")
    # the BuildConfig next to the current one, and a .build.info row
    cfg_dir = os.path.join(install, "SC2Data", "config", p["bc_hash"][:2], p["bc_hash"][2:4])
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, p["bc_hash"]), "wb") as f:
        f.write(cdn.cached("config", p["bc_hash"]))
    if not row_present:
        row = dict(p["active"], Active="0", **{"Build Key": p["bc_hash"], "Version": build_version(b, p["build"])})
        cols = [c.split("!")[0] for c in p["header"].split("|")]
        with open(os.path.join(install, ".build.info"), "a") as f:
            f.write("|".join(row.get(c, "") for c in cols) + "\n")
        log(f".build.info: added row for {row['Version']} (Active 0)")
    log(f"build {p['build']} installed in {time.time() - t0:.0f} s")
    return p


GAME_VERSIONS = {93333: "5.0.14", 94137: "5.0.14", 95841: "5.0.15", 97364: "5.0.16", 97563: "5.0.16", 98310: "5.0.17"}


def build_version(b, build):
    """The .build.info Version string, e.g. 5.0.14.93333 (the game version is not in the
    BuildConfig; unknown builds get 5.0.x)."""
    del b
    return f"{GAME_VERSIONS.get(build, '5.0.x')}.{build}"


def storage_read(storage, ekey):
    arch, off, size = storage.entries.get(ekey[:9]) or storage.pending[ekey[:9]]
    with open(os.path.join(storage.dir, f"data.{arch:03d}"), "rb") as f:
        f.seek(off + 0x1E)
        return f.read(size - 0x1E)


def main(argv=None):
    import argparse
    from cosmicstar.envs.wine_run_config import sc2_install_dir
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("build", help="base build (e.g. 93333) or <build>=<BuildConfig hash>")
    ap.add_argument("--install", default=sc2_install_dir(), help="the StarCraft II folder")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="where configs, manifests and indexes are cached")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="only report what is missing")
    ap.add_argument("--force", action="store_true", help="write the storage even while the engine runs")
    ap.add_argument("--revert", metavar="BACKUP_DIR", help="undo the install of <build> from this SC2Data/backup-* folder")
    a = ap.parse_args(argv)
    build = int(a.build) if a.build.isdigit() else a.build
    if a.revert:
        return revert(a.install, a.revert, build if isinstance(build, int) else None)
    install_build(a.install, build, dry_run=a.dry_run, cache=a.cache, threads=a.threads, force=a.force)


if __name__ == "__main__":
    main()
