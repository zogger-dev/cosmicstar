"""Blizzard's StarCraft II depot: maps and mods by content hash.

A replay's `replay.details` lists cache handles, one per map or mod the game used: 4 bytes of
file type (b"s2ma" map, b"s2mv" map preview, b"s2mh" map header, b"s2ml" locale, ...), 4 bytes
of region (b"us\\0\\0") and the file's 32-byte SHA-256. The client resolves a handle to
    https://<region>-s2-depot.classic.blizzard.com/<sha256 hex>.<type>
and stores it in the Battle.net cache as <Cache>/<hex[:2]>/<hex[2:4]>/<hex>.<type>. That URL
template is in SC2_x64.exe (strings: "-s2-depot.classic.blizzard.com"); the older
http://<region>.depot.battle.net:1119/ hosts, which the usual tools use, stopped resolving
in 2026. The API client does not fetch missing files itself (ResponseReplayInfo
ParsingError: "Failed to open the map archive"), only the game UI does, so anything that
plays replays from the API has to fill the cache first: fetch() / ensure_cached() below.
"""
import os
import tempfile

DEPOT_URL = "https://{region}-s2-depot.classic.blizzard.com/{hash}.{ext}"
REGIONS = ("us", "eu", "kr")  # the depot is global; the host only picks a CDN edge
MPQ_TYPES = {"s2ma", "s2mv"}  # types that are MPQ archives (magic "MPQ\\x1a")


def parse_handle(handle):
    """(ext, hex hash) of a replay cache handle (bytes from replay.details m_cacheHandles)."""
    ext = handle[:4].decode("ascii").rstrip("\0")
    return ext, handle[8:].hex()


def cache_path(cache_dir, hash_hex, ext):
    return os.path.join(cache_dir, hash_hex[:2], hash_hex[2:4], f"{hash_hex}.{ext}")


def find_cached(cache_dirs, hash_hex, ext):
    for d in cache_dirs:
        p = cache_path(d, hash_hex, ext)
        if os.path.exists(p):
            return p
    return None


def fetch(hash_hex, ext, cache_dir, region="us", timeout=180):
    """Downloads one depot file into the cache (atomically) and returns its path; raises on
    HTTP errors or a non-MPQ body for archive types."""
    import requests
    url = DEPOT_URL.format(region=region, hash=hash_hex, ext=ext)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    if ext in MPQ_TYPES and r.content[:4] != b"MPQ\x1a":
        raise ValueError(f"{url}: not an MPQ archive (starts {r.content[:8]!r})")
    dest = cache_path(cache_dir, hash_hex, ext)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), prefix=".fetch-")
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    return dest


def ensure_cached(handles, cache_dirs, log=None):
    """Fetches every handle (bytes, or (ext, hex) pairs) missing from the cache into
    cache_dirs[0]. Returns the list of (ext, hex) it could not get."""
    missing = []
    for h in handles:
        ext, hx = parse_handle(h) if isinstance(h, (bytes, bytearray)) else h
        if find_cached(cache_dirs, hx, ext):
            continue
        try:
            p = fetch(hx, ext, cache_dirs[0])
            if log:
                log(f"fetched {ext} {hx[:12]} ({os.path.getsize(p) / 1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001 -- report and keep going; the caller decides
            missing.append((ext, hx))
            if log:
                log(f"depot fetch failed for {ext} {hx[:12]}: {e}")
    return missing
