"""Broadcast HUD, production cards, upgrades, and `.SC2Replay` combat casualty extraction."""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import Any

import mpyq
import numpy as np
import torch

from ..core.constants import P1, P2, PC
from ..data import statefile
from ..data.game_store import NEUTRAL, QUEUE
from ..l1_tactical.frames import I
from .map_s2ma import clean_player_name

logger = logging.getLogger(__name__)


class ReplayHudMixin:
    """Broadcast HUD, casualty, resource-loss, upgrade, and production card methods."""

    def _get_game_cst_stats(self, gid: str) -> dict[str, Any] | None:
        if gid in self._cst_cache:
            return self._cst_cache[gid]
        try:
            g = self._get_game(gid)
            cst_path = g.base + ".p1.cst"
            stats = {}
            if os.path.exists(cst_path):
                with statefile.load(cst_path) as d:
                    stats = {
                        "p_loop": d["p_loop"].copy() if "p_loop" in d.files else None,
                        "p_player": d["p_player"].copy() if "p_player" in d.files else None,
                        "p_mineral_rate": d["p_mineral_rate"].copy() if "p_mineral_rate" in d.files else None,
                        "p_vespene_rate": d["p_vespene_rate"].copy() if "p_vespene_rate" in d.files else None,
                        "p_minerals_lost_army": d["p_minerals_lost_army"].copy() if "p_minerals_lost_army" in d.files else None,
                        "p_minerals_lost_economy": d["p_minerals_lost_economy"].copy() if "p_minerals_lost_economy" in d.files else None,
                        "p_minerals_lost_technology": d["p_minerals_lost_technology"].copy() if "p_minerals_lost_technology" in d.files else None,
                        "p_vespene_lost_army": d["p_vespene_lost_army"].copy() if "p_vespene_lost_army" in d.files else None,
                        "p_vespene_lost_economy": d["p_vespene_lost_economy"].copy() if "p_vespene_lost_economy" in d.files else None,
                        "p_vespene_lost_technology": d["p_vespene_lost_technology"].copy() if "p_vespene_lost_technology" in d.files else None,
                        "dead_frame_start": d["dead_frame_start"].copy() if "dead_frame_start" in d.files else None,
                        "dead_id": d["dead_id"].copy() if "dead_id" in d.files else None,
                        "up_loop": d["up_loop"].copy() if "up_loop" in d.files else None,
                        "up_player": d["up_player"].copy() if "up_player" in d.files else None,
                        "up_name": d["up_name"].copy() if "up_name" in d.files else None,
                    }
                unit_map = {}
                arrow_path = g.base + ".p1.units.arrow"
                if os.path.exists(arrow_path):
                    import pyarrow.feather as feather
                    table = feather.read_table(arrow_path, columns=["u_id", "u_type", "u_owner"])
                    u_ids = table.column("u_id").to_numpy(zero_copy_only=False)
                    u_types = table.column("u_type").to_numpy(zero_copy_only=False)
                    u_owners = table.column("u_owner").to_numpy(zero_copy_only=False)
                    _, idx = np.unique(u_ids, return_index=True)
                    unit_map = {int(u_ids[i]): (int(u_types[i]), int(u_owners[i])) for i in idx}
                stats["unit_map"] = unit_map

            # Parse genuine combat casualties from replay tracker events if replay is available
            kills = None
            replay_path = self._replay_path_map.get(gid)
            if replay_path and os.path.exists(replay_path):
                try:
                    import s2protocol.versions
                    archive = mpyq.MPQArchive(replay_path)
                    header_content = archive.header["user_data_header"]["content"]
                    header = s2protocol.versions.latest().decode_replay_header(header_content)
                    base_build = header["m_version"]["m_baseBuild"]
                    protocol = s2protocol.versions.build(base_build)
                    contents = archive.read_file("replay.tracker.events")
                    events = list(protocol.decode_replay_tracker_events(contents))

                    unit_dict = {}
                    kills_list = []
                    for e in events:
                        ev = e["_event"]
                        idx = e.get("m_unitTagIndex")
                        rec = e.get("m_unitTagRecycle")
                        key = (idx, rec)
                        if ev in ("NNet.Replay.Tracker.SUnitBornEvent", "NNet.Replay.Tracker.SUnitInitEvent"):
                            unit_dict[key] = {
                                "type": e["m_unitTypeName"].decode("utf-8", errors="ignore"),
                                "owner": e["m_controlPlayerId"],
                            }
                        elif ev == "NNet.Replay.Tracker.SUnitTypeChangeEvent":
                            if key in unit_dict:
                                unit_dict[key]["type"] = e["m_unitTypeName"].decode("utf-8", errors="ignore")
                        elif ev == "NNet.Replay.Tracker.SUnitDiedEvent":
                            u_info = unit_dict.get(key)
                            if not u_info:
                                continue
                            u_own = u_info["owner"]
                            u_type = u_info["type"]
                            if u_own not in (1, 2):
                                continue
                            if u_type.lower() in self.exclude_from_dead:
                                continue
                            k_pid = e.get("m_killerPlayerId")
                            k_idx = e.get("m_killerUnitTagIndex")
                            k_rec = e.get("m_killerUnitTagRecycle")
                            # Exclude morphs (k_pid is None: Drone morphing to building, Larva to egg, HT to Archon, etc.)
                            if k_pid is None:
                                continue

                            loop_k = e["_gameloop"]
                            is_cancel = (k_idx == idx and k_rec == rec)

                            if is_cancel:
                                # When a building is cancelled, 75% is refunded, losing 25% of building cost
                                u_lower = u_type.lower()
                                if u_lower in self._structure_names or u_lower in self.zerg_build_costs:
                                    base_c = self.zerg_build_costs.get(u_lower) or self._unit_costs.get(u_lower, (0, 0))
                                    mc = math.ceil(base_c[0] * 0.25)
                                    vc = math.ceil(base_c[1] * 0.25)
                                    kills_list.append((loop_k, u_own, u_type, mc, vc, True))
                                elif u_lower == "baneling":
                                    # Baneling self-detonation in combat is a 100% combat loss
                                    c = self._unit_costs.get(u_lower, (50, 25))
                                    kills_list.append((loop_k, u_own, u_type, c[0], c[1], False))
                            else:
                                costs = self._unit_costs.get(u_type.lower(), (0, 0))
                                kills_list.append((loop_k, u_own, u_type, costs[0], costs[1], False))
                    kills_list.sort(key=lambda x: x[0])
                    kills = kills_list
                except Exception as ex:
                    logger.warning("Failed to decode tracker events for %s: %s", gid, ex)

            stats["kills"] = kills
            self._cst_cache[gid] = stats
            return stats
        except Exception:
            self._cst_cache[gid] = None
            return None

    SPECIAL_PROD_MAP: dict[str, str] = {
        "orbital": "OrbitalCommand",
        "upgradetoorbital": "OrbitalCommand",
        "orbitalcommand": "OrbitalCommand",
        "planetary": "PlanetaryFortress",
        "planetaryfortress": "PlanetaryFortress",
        "upgradetoplanetaryfortress": "PlanetaryFortress",
        "upgradetowarpgate": "WarpGate",
        "warpgate": "WarpGate",
        "upgradetolair": "Lair",
        "lair": "Lair",
        "upgradetohive": "Hive",
        "hive": "Hive",
        "upgradetogreaterspire": "GreaterSpire",
        "greaterspire": "GreaterSpire",
        "morphtotransportoverlord": "OverlordTransport",
        "morphtohelliontank": "Hellbat",
        "morphtolurker": "Lurker",
        "morphtoravager": "Ravager",
        "morphtobaneling": "Baneling",
        "morphtobroodlord": "BroodLord",
        "morphtodevourermp": "Devourer",
        "morphtoguardianmp": "Guardian",
        "morphtooverseer": "Overseer",
        "morphtomothership": "Mothership",
        "barracksaddons": "TechLab",
        "barrackstechlab": "TechLab",
        "barracksreactor": "Reactor",
        "factorytechlab": "TechLab",
        "factoryreactor": "Reactor",
        "starporttechlab": "TechLab",
        "starportreactor": "Reactor",
    }

    @classmethod
    def _parse_prod_abil_name(cls, ab_str: str) -> tuple[str, str]:
        if not ab_str or ab_str in ("None", "Unknown"):
            return "", "unit"
        parts = ab_str.split("/")
        last = parts[-1]
        first = parts[0]

        # If last part is an action keyword like Execute, Cancel, None
        if last.lower() in ("execute", "cancel", "none", "halt", "on", "off"):
            cand = first
        else:
            cand = last

        cand_lower = cand.lower()
        if cand_lower in ("execute", "cancel", "none", "halt", "on", "off", "stop", "holdfire"):
            return "", "unit"
        if cand_lower in cls.SPECIAL_PROD_MAP:
            name = cls.SPECIAL_PROD_MAP[cand_lower]
        else:
            name = cand
            for pfx in ("UpgradeTo", "MorphTo", "Build", "Train", "Research"):
                if name.startswith(pfx) and len(name) > len(pfx):
                    name = name[len(pfx):]
                    break
            if name.lower() in cls.SPECIAL_PROD_MAP:
                name = cls.SPECIAL_PROD_MAP[name.lower()]

        if name.lower() in ("execute", "cancel", "none", "halt", "on", "off", "stop", "holdfire"):
            return "", "unit"

        low_name = name.lower()
        low_ab = ab_str.lower()
        if "research" in low_ab or any(u in low_name for u in ["weapons", "armors", "plating", "shield", "speed", "stimpack", "boost", "glaives"]):
            cat = "upgrade"
        elif any(s in low_name for s in ["orbitalcommand", "planetaryfortress", "lair", "hive", "greaterspire", "warpgate", "techlab", "reactor"]):
            cat = "structure"
        else:
            cat = "unit"
        return name, cat

    def _extract_hud_and_production(
        self,
        g: Game,
        w: dict[str, Any],
        b: dict[str, torch.Tensor],
        f_curr: int,
        loop_t: int,
        units_out: list[dict[str, Any]],
        meta_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extracts live match observer HUD (minerals, vespene, income rates, supply used/cap,
        workers, army, completed upgrades), active production items (with progress bars and counts),
        and forces grouped by type with unit row IDs."""
        pl = b["ply_num"].float()
        v = torch.expm1(pl * 8.0)

        meta = meta_info or (g.meta if hasattr(g, "meta") else {})
        p1_name = clean_player_name(meta.get("p1_name", "Player 1"))
        p2_name = clean_player_name(meta.get("p2_name", "Player 2"))
        p1_race = meta.get("p1_race", "Terran")
        p2_race = meta.get("p2_race", "Zerg")

        # 1. Load CST stats for income rates & upgrades & losses (cached per game)
        gid_str = meta.get("gid", os.path.basename(g.base))
        cst_stats = self._get_game_cst_stats(gid_str)
        min_rate = [0, 0]
        vesp_rate = [0, 0]
        min_lost = [0, 0]
        vesp_lost = [0, 0]
        dead_groups = {1: defaultdict(int), 2: defaultdict(int)}
        p1_ups, p2_ups = [], []
        if cst_stats is not None:
            p_l = cst_stats.get("p_loop")
            p_p = cst_stats.get("p_player")
            mr = cst_stats.get("p_mineral_rate")
            vr = cst_stats.get("p_vespene_rate")
            if p_l is not None and p_p is not None:
                m1 = (p_l <= loop_t) & (p_p == 1)
                m2 = (p_l <= loop_t) & (p_p == 2)
                if m1.any():
                    idx1 = np.where(m1)[0][-1]
                    if mr is not None:
                        min_rate[0] = int(mr[idx1])
                    if vr is not None:
                        vesp_rate[0] = int(vr[idx1])
                if m2.any():
                    idx2 = np.where(m2)[0][-1]
                    if mr is not None:
                        min_rate[1] = int(mr[idx2])
                    if vr is not None:
                        vesp_rate[1] = int(vr[idx2])

            kills = cst_stats.get("kills")
            if kills is not None:
                for entry in kills:
                    if len(entry) == 6:
                        loop_k, own, nm, mc, vc, is_cancel = entry
                    else:
                        loop_k, own, nm, mc, vc = entry
                        is_cancel = False
                    if loop_k <= loop_t:
                        min_lost[own - 1] += mc
                        vesp_lost[own - 1] += vc
                        if not is_cancel:
                            dead_groups[own][nm] += 1
            else:
                # Fallback to CST stats if tracker events unavailable
                mla = cst_stats.get("p_minerals_lost_army")
                mle = cst_stats.get("p_minerals_lost_economy")
                mlt = cst_stats.get("p_minerals_lost_technology")
                vla = cst_stats.get("p_vespene_lost_army")
                vle = cst_stats.get("p_vespene_lost_economy")
                vlt = cst_stats.get("p_vespene_lost_technology")
                if p_l is not None and p_p is not None:
                    m1 = (p_l <= loop_t) & (p_p == 1)
                    m2 = (p_l <= loop_t) & (p_p == 2)
                    if m1.any():
                        idx1 = np.where(m1)[0][-1]
                        if mla is not None and mle is not None and mlt is not None:
                            min_lost[0] = max(0, int(mla[idx1] + mle[idx1] + mlt[idx1]))
                        if vla is not None and vle is not None and vlt is not None:
                            vesp_lost[0] = max(0, int(vla[idx1] + vle[idx1] + vlt[idx1]))
                    if m2.any():
                        idx2 = np.where(m2)[0][-1]
                        if mla is not None and mle is not None and mlt is not None:
                            min_lost[1] = max(0, int(mla[idx2] + mle[idx2] + mlt[idx2]))
                        if vla is not None and vle is not None and vlt is not None:
                            vesp_lost[1] = max(0, int(vla[idx2] + vle[idx2] + vlt[idx2]))

                dfs = cst_stats.get("dead_frame_start")
                did = cst_stats.get("dead_id")
                unit_map = cst_stats.get("unit_map", {})
                if dfs is not None and did is not None and len(dfs) > 0 and len(did) > 0:
                    l_idx = min(max(0, loop_t), len(dfs) - 1)
                    dead_cnt = int(dfs[l_idx])
                    dead_slice = did[:dead_cnt]
                    for uid in dead_slice:
                        info = unit_map.get(int(uid))
                        if info:
                            utype, uown = info
                            if uown in (1, 2):
                                um = self.unit_meta.get(utype, {})
                                nm = um.get("name", f"Unit_{utype}")
                                if nm.lower() not in self.exclude_from_dead:
                                    dead_groups[uown][nm] += 1

            u_l = cst_stats.get("up_loop")
            u_p = cst_stats.get("up_player")
            u_n = cst_stats.get("up_name")
            if u_l is not None and u_p is not None and u_n is not None:
                p1_ups = [str(n) for l, p, n in zip(u_l, u_p, u_n) if l <= loop_t and p == 1]
                p2_ups = [str(n) for l, p, n in zip(u_l, u_p, u_n) if l <= loop_t and p == 2]

        def _format_upgrades(raw_list):
            cleaned = {}
            for raw in raw_list:
                name = raw
                lvl = 1
                if "Level1" in raw:
                    name = raw.replace("Level1", ""); lvl = 1
                elif "Level2" in raw:
                    name = raw.replace("Level2", ""); lvl = 2
                elif "Level3" in raw:
                    name = raw.replace("Level3", ""); lvl = 3
                if name not in cleaned or lvl > cleaned[name]["level"]:
                    cleaned[name] = {"name": name, "raw": raw, "level": lvl}
            return list(cleaned.values())

        hud = {
            "p1": {
                "name": p1_name,
                "race": p1_race,
                "minerals": int(v[0, f_curr, 0, PC["minerals"]].item()),
                "vespene": int(v[0, f_curr, 0, PC["vespene"]].item()),
                "minerals_rate": min_rate[0],
                "vespene_rate": vesp_rate[0],
                "supply_used": int(v[0, f_curr, 0, PC["food_used"]].item()),
                "supply_cap": int(v[0, f_curr, 0, PC["food_cap"]].item()),
                "workers": int(v[0, f_curr, 0, PC["food_workers"]].item()),
                "idle_workers": int(v[0, f_curr, 0, PC["idle_worker_count"]].item()),
                "army_supply": int(v[0, f_curr, 0, PC["food_army"]].item()),
                "upgrades": _format_upgrades(p1_ups),
                "minerals_lost": min_lost[0],
                "vespene_lost": vesp_lost[0],
                "resources_lost": min_lost[0] + vesp_lost[0],
            },
            "p2": {
                "name": p2_name,
                "race": p2_race,
                "minerals": int(v[0, f_curr, 1, PC["minerals"]].item()),
                "vespene": int(v[0, f_curr, 1, PC["vespene"]].item()),
                "minerals_rate": min_rate[1],
                "vespene_rate": vesp_rate[1],
                "supply_used": int(v[0, f_curr, 1, PC["food_used"]].item()),
                "supply_cap": int(v[0, f_curr, 1, PC["food_cap"]].item()),
                "workers": int(v[0, f_curr, 1, PC["food_workers"]].item()),
                "idle_workers": int(v[0, f_curr, 1, PC["idle_worker_count"]].item()),
                "army_supply": int(v[0, f_curr, 1, PC["food_army"]].item()),
                "upgrades": _format_upgrades(p2_ups),
                "minerals_lost": min_lost[1],
                "vespene_lost": vesp_lost[1],
                "resources_lost": min_lost[1] + vesp_lost[1],
            },
        }

        # 2. Production queue (actively building slot 0 only!)
        pw_list = w.get("prod_where", [])
        pv_list = w.get("prod_vals", [])
        EXCLUDE_FROM_FORCES = {
            "egg", "larva",
            "banelingcocoon", "ravagercocoon",
            "lurkeregg", "lurkermpcocoon",
            "broodlordcocoon", "overlordcocoon", "transportoverlordcocoon",
        }
        unit_by_row = {u["row"]: u for u in units_out if "row" in u}
        prod_groups = {
            1: defaultdict(lambda: {"count": 0, "max_prog": 0.0, "category": "unit"}),
            2: defaultdict(lambda: {"count": 0, "max_prog": 0.0, "category": "unit"}),
        }

        # Iterate all production chunks and assign each order to its producer's true owner
        for wh, vl in zip(pw_list, pv_list):
            if wh is None or vl is None or len(wh) == 0:
                continue
            mask = (wh[:, 0] == f_curr) & (wh[:, 2] == 0)
            for i in np.where(mask)[0]:
                r = int(wh[i, 1])
                owner = unit_by_row.get(r, {}).get("owner")
                if owner is None and "owner" in b and r < b["owner"].shape[1]:
                    raw_own = int(b["owner"][0, r].item())
                    owner = 1 if raw_own == 0 else (2 if raw_own == 1 else 0)
                if owner not in (1, 2):
                    continue
                raw_abil = int(vl[i, 0])
                prog = float(vl[i, 1])
                tok = self.vocab._abil.get(raw_abil) if hasattr(self.vocab, "_abil") else None
                ab_str = self.vocab_abil_name.get(tok, "") if tok is not None else ""
                raw_name, cat = self._parse_prod_abil_name(ab_str)
                if not raw_name:
                    continue
                mult = 2 if raw_name.lower() == "zergling" else 1
                prod_groups[owner][raw_name]["count"] += mult
                prod_groups[owner][raw_name]["max_prog"] = max(prod_groups[owner][raw_name]["max_prog"], prog)
                prod_groups[owner][raw_name]["category"] = cat

        # Also check structures currently under construction and morphing cocoons/eggs
        for u in units_out:
            own = u.get("owner")
            if own not in (1, 2):
                continue
            if u.get("structure"):
                r = u.get("row")
                if r is not None and r < b["num"].shape[2]:
                    bld = float(b["num"][0, f_curr, r, I["build"]].item())
                    if bld < 0.99:
                        sname = u["name"]
                        prod_groups[own][sname]["count"] += 1
                        prod_groups[own][sname]["max_prog"] = max(prod_groups[own][sname]["max_prog"], bld)
                        prod_groups[own][sname]["category"] = "structure"

            if u.get("name", "").lower() in EXCLUDE_FROM_FORCES:
                target_abil = u.get("prod_abil") or u.get("order_abil")
                if target_abil and target_abil not in ("None", "Unknown"):
                    t_name, t_cat = self._parse_prod_abil_name(target_abil)
                    if t_name and t_name not in prod_groups[own]:
                        r = u.get("row")
                        bld = 0.0
                        if r is not None and r < b["num"].shape[2]:
                            bld = float(b["num"][0, f_curr, r, I["build"]].item())
                        mult = 2 if t_name.lower() == "zergling" else 1
                        prod_groups[own][t_name]["count"] += mult
                        prod_groups[own][t_name]["max_prog"] = max(prod_groups[own][t_name]["max_prog"], bld)
                        prod_groups[own][t_name]["category"] = t_cat

        prod = {"p1": [], "p2": []}
        for own in (1, 2):
            pk = f"p{own}"
            items = []
            for k, val in prod_groups[own].items():
                items.append({
                    "name": k,
                    "count": val["count"],
                    "progress": round(val["max_prog"], 3),
                    "category": val["category"],
                })
            prod[pk] = sorted(items, key=lambda x: -x["progress"])

        # 3. Forces & Structures breakdown
        forces = {
            "p1": {"army": [], "workers": [], "structures": []},
            "p2": {"army": [], "workers": [], "structures": []},
        }
        f_groups = {
            1: {"army": defaultdict(list), "workers": defaultdict(list), "structures": defaultdict(list)},
            2: {"army": defaultdict(list), "workers": defaultdict(list), "structures": defaultdict(list)},
        }
        for u in units_out:
            own = u.get("owner")
            if own in (1, 2):
                nm = u["name"]
                if nm.lower() in EXCLUDE_FROM_FORCES:
                    continue
                if u.get("worker"):
                    f_groups[own]["workers"][nm].append(u["row"])
                elif u.get("structure"):
                    f_groups[own]["structures"][nm].append(u["row"])
                else:
                    f_groups[own]["army"][nm].append(u["row"])

        for own in (1, 2):
            pk = f"p{own}"
            for cat in ("army", "workers", "structures"):
                items = []
                for nm, rows in f_groups[own][cat].items():
                    items.append({
                        "name": nm,
                        "count": len(rows),
                        "rows": rows,
                    })
                forces[pk][cat] = sorted(items, key=lambda x: -x["count"])

        # 4. Casualties & Lost Units
        units_lost = {"p1": [], "p2": []}
        for own in (1, 2):
            pk = f"p{own}"
            items = []
            for nm, cnt in dead_groups[own].items():
                items.append({
                    "name": nm,
                    "count": cnt,
                })
            units_lost[pk] = sorted(items, key=lambda x: -x["count"])

        return {
            "hud": hud,
            "production": prod,
            "forces": forces,
            "units_lost": units_lost,
        }
