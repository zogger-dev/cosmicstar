"""SC2 `.s2ma` map archive cache, player name sanitizer, and 3D PCA RGB helper."""

from __future__ import annotations

import glob
import html
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import mpyq
import numpy as np


def clean_player_name(raw: Any) -> str:
    """Cleans StarCraft II player name by decoding HTML entities, removing
    Blizzard <sp/> tags, and stripping formatting tags like <c val="...">."""
    if not raw:
        return ""
    s = html.unescape(str(raw))
    # Replace SC2 XML space tags like <sp/> or <sp> with a single space
    s = re.sub(r"<\s*/?\s*sp\s*/?>", " ", s, flags=re.I)
    # Strip SC2 color/style tags like <c val="..."> and </c>
    s = re.sub(r"<\/?c(\s+[^>]*)?>", "", s, flags=re.I)
    # Collapse multiple consecutive whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pca_rgb(vecs: np.ndarray) -> np.ndarray:
    """Projects [N, D] vectors onto their top 3 principal components and normalizes to [0, 255] RGB."""
    N, D = vecs.shape
    if N == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if N < 3:
        out = np.full((N, 3), 160, dtype=np.uint8)
        return out
    centered = vecs - vecs.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[:3].T  # [N, 3]
        lo = np.percentile(proj, 5, axis=0, keepdims=True)
        hi = np.percentile(proj, 95, axis=0, keepdims=True)
        normed = np.clip((proj - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
        return (normed * 215.0 + 40.0).astype(np.uint8)
    except Exception:
        return np.full((N, 3), 160, dtype=np.uint8)


class S2MAMapCache:
    """Parses .s2ma MPQ map archives from the Battle.net Cache to extract exact
    NoFlyZone air blockers, XelNagaHealingShrines, and special terrain objects."""

    def __init__(self, cache_glob: str | None = None):
        if cache_glob is None:
            cache_glob = os.path.expanduser(
                "~/Games/battlenet/drive_c/ProgramData/Blizzard Entertainment/Battle.net/Cache/*/*/*.s2ma"
            )
        self.by_hash: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        self._parsed: dict[str, dict[str, Any]] = {}

        for p in glob.glob(cache_glob):
            h = os.path.splitext(os.path.basename(p))[0]
            self.by_hash[h] = p
            self.by_hash[h[:16]] = p
            try:
                a = mpyq.MPQArchive(p, listfile=False)
                dh = a.read_file("DocumentHeader")
                if dh:
                    strings = [s.decode("utf-8", errors="ignore") for s in re.findall(b"[ -~]{4,}", dh)]
                    for idx, s in enumerate(strings):
                        if "DocInfo/Name" in s and idx + 1 < len(strings):
                            name = strings[idx + 1].strip()
                            self.by_name[name] = p
                            break
            except Exception:
                continue

    def get_map_objects(self, map_hash: str, map_name: str) -> dict[str, Any]:
        p = (
            self.by_hash.get(map_hash)
            or self.by_hash.get(map_hash[:16])
            or self.by_name.get(map_name)
        )
        if not p:
            return {"found_s2ma": False, "no_fly_zones": [], "healing_shrines": [], "objects": []}
        if p in self._parsed:
            return self._parsed[p]

        no_fly_zones = []
        healing_shrines = []
        other_objs = []
        try:
            a = mpyq.MPQArchive(p, listfile=False)
            objs_raw = a.read_file("Objects")
            if objs_raw:
                root = ET.fromstring(objs_raw.decode("utf-8", errors="ignore"))
                for el in root:
                    pos_str = el.attrib.get("Position", "")
                    parts = [float(x) for x in pos_str.split(",") if x.strip()]
                    if len(parts) < 2:
                        continue
                    x, y = parts[0], parts[1]
                    z = parts[2] if len(parts) > 2 else 0.0
                    t = el.attrib.get("Type", "") or el.attrib.get("UnitType", "")
                    if t == "NoFlyZone":
                        r_hard = float(el.attrib.get("PathingRadiusHard") or 3.0)
                        r_soft = float(el.attrib.get("PathingRadiusSoft") or r_hard)
                        no_fly_zones.append(
                            {
                                "x": round(x, 2),
                                "y": round(y, 2),
                                "z": round(z, 2),
                                "r_hard": round(r_hard, 2),
                                "r_soft": round(r_soft, 2),
                            }
                        )
                    elif "HealingShrine" in t:
                        healing_shrines.append(
                            {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "type": t}
                        )
                    elif any(
                        k in t
                        for k in (
                            "InhibitorZone",
                            "AccelerationZone",
                            "XelNagaTower",
                            "Destructible",
                            "Unbuildable",
                        )
                    ):
                        other_objs.append(
                            {"x": round(x, 2), "y": round(y, 2), "type": t}
                        )
        except Exception:
            pass

        res = {
            "found_s2ma": True,
            "s2ma_path": os.path.basename(p),
            "no_fly_zones": no_fly_zones,
            "healing_shrines": healing_shrines,
            "objects": other_objs[:120],
        }
        self._parsed[p] = res
        return res
