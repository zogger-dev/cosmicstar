"""Parses a replay's game events into a per-player action table (no game engine needed).

Every player's interface actions, at the exact game loop, one row per event:

    kind        CMD           a command: ability (or right-click), flags, target
                CMD_POINT     re-target of the previous command to a point (spam clicks)
                CMD_UNIT      re-target of the previous command to a unit
                REPEAT        "repeat the last command" (command-manager state)
                SELECT        selection change: units added, units removed
                GROUP         control group set / add / recall / clear (/ steal)
                CAMERA        camera moved to a point
                LEAVE         the player left the game
                CAMERA_SAVE   camera hotkey saved (group = slot, cam_x/cam_y = position);
                              a recall shows up as a plain CAMERA move to that position
                PING          minimap/world ping (tx/ty; target_tag/snap_link if a unit was
                              pinged; flags has FLAG_MINIMAP if pinged on the minimap)
                TRADE         resources sent to a teammate (recipient, minerals, gas)
                REQUEST       resources requested from teammates (minerals, gas)
                REQUEST_FULFILL / REQUEST_CANCEL   (request_id)

Fields (not every field applies to every kind; -1 / NaN / 0 = not applicable):

    loop, player              game loop, player slot (1-based, as in the state's meta)
    abil_link, abil_cmd       the replay's own ability numbering (translated to API ids later)
    flags                     command flags: 0x2 queued, 0x8 smart click (right-click),
                              0x100 user, 0x10000 minimap, 0x20000 repeat, ...
    target                    TARGET_NONE / POINT / UNIT / SNAPSHOT (a fogged thing: no tag)
    tx, ty                    target position in map coordinates (points, units, snapshots)
    target_tag                replay tag of the target unit (engine tag = 2**32 + replay tag)
    snap_link, snap_player    for snapshots: the replay's unit-type link and its owner
    group                     control group (0-9); 10 = the active selection (SELECT rows)
    group_op                  GROUP rows: 0 set, 1 add, 2 recall, 3 clear, 4 set+steal, 5 add+steal
    mgr_state, sequence       REPEAT rows; command sequence numbers
    cam_x, cam_y              CAMERA rows: where the camera moved
    select_remove             SELECT rows: REMOVE_NONE / MASK / ONE (positions to remove) /
                              ZERO (positions to keep, remove the rest)
    recipient                 TRADE rows: receiving player slot
    minerals, gas             TRADE / REQUEST rows: amounts
    request_id                REQUEST_FULFILL / REQUEST_CANCEL rows
    rem_*, add_*, addsub_*    ragged lists per row: removal positions, added unit tags, added
                              per-type counts (unit link, count); row i owns
                              [<x>_start[i], <x>_start[i+1])

Unit references are raw replay tags; the tokenizer maps them to state ids through the
state file's unit_tag table.
"""

import json

import numpy as np

(CMD, CMD_POINT, CMD_UNIT, REPEAT, SELECT, GROUP, CAMERA, LEAVE,
 CAMERA_SAVE, PING, TRADE, REQUEST, REQUEST_FULFILL, REQUEST_CANCEL) = range(14)
KINDS = ("CMD", "CMD_POINT", "CMD_UNIT", "REPEAT", "SELECT", "GROUP", "CAMERA", "LEAVE",
         "CAMERA_SAVE", "PING", "TRADE", "REQUEST", "REQUEST_FULFILL", "REQUEST_CANCEL")
TARGET_NONE, TARGET_POINT, TARGET_UNIT, TARGET_SNAPSHOT = range(4)
REMOVE_NONE, REMOVE_MASK, REMOVE_ONE, REMOVE_ZERO = range(4)
FLAG_QUEUED, FLAG_SMART, FLAG_MINIMAP, FLAG_REPEAT = 0x2, 0x8, 0x10000, 0x20000
POINT_SCALE, CAMERA_SCALE = 4096.0, 256.0

COLUMNS = dict(loop=np.int32, player=np.int8, kind=np.int8, abil_link=np.int16, abil_cmd=np.int16,
               flags=np.int64, target=np.int8, tx=np.float32, ty=np.float32, target_tag=np.int64,
               snap_link=np.int32, snap_player=np.int8, group=np.int8, group_op=np.int8, mgr_state=np.int8,
               sequence=np.int32, cam_x=np.float32, cam_y=np.float32, select_remove=np.int8,
               recipient=np.int8, minerals=np.int32, gas=np.int32, request_id=np.int32,
               rem_start=np.int64, add_start=np.int64, addsub_start=np.int64)
RAGGED = dict(rem_pos=np.int32, add_tag=np.int64, addsub_link=np.int32, addsub_count=np.int16)


def _protocol(archive):
    from s2protocol import versions
    header = versions.latest().decode_replay_header(archive.header["user_data_header"]["content"])
    build = header["m_version"]["m_build"]
    try:
        return versions.build(build), build
    except ImportError:
        return versions.latest(), build


def _s(b):
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else b


def user_slots(proto, archive):
    """Lobby user id -> player slot (1-based index in the details' player list)."""
    det = proto.decode_replay_details(archive.read_file("replay.details"))
    init = proto.decode_replay_initdata(archive.read_file("replay.initData"))["m_syncLobbyState"]
    slot_user = {sl["m_workingSetSlotId"]: sl["m_userId"] for sl in init["m_lobbyState"]["m_slots"]
                 if sl.get("m_userId") is not None}
    out = {}
    for i, p in enumerate(det["m_playerList"]):
        u = slot_user.get(p.get("m_workingSetSlotId"))
        if u is not None:
            out[u] = i + 1
    return out, [dict(slot=i + 1, name=_s(p["m_name"]), race=_s(p["m_race"]), team=p["m_teamId"])
                 for i, p in enumerate(det["m_playerList"])]


def _mask_positions(bits):
    """s2protocol bitmask (length, value) -> positions of the set bits."""
    n, v = bits
    return [i for i in range(n) if (v >> i) & 1]


def parse(path):
    """Replay file -> dict of numpy arrays (see the module doc) plus a meta JSON string."""
    import mpyq
    a = mpyq.MPQArchive(path)
    proto, build = _protocol(a)
    slots, players = user_slots(proto, a)
    rows = {k: [] for k in COLUMNS}
    rag = {k: [] for k in RAGGED}
    skipped = {}

    def row(loop, player, kind, **kw):
        for k in COLUMNS:
            if k in ("rem_start", "add_start", "addsub_start"):
                continue
            default = np.nan if k in ("tx", "ty", "cam_x", "cam_y") else (0 if k in ("flags", "target_tag") else -1)
            rows[k].append(kw.get(k, default))
        rows["loop"][-1], rows["player"][-1], rows["kind"][-1] = loop, player, kind
        rows["rem_start"].append(len(rag["rem_pos"]))
        rows["add_start"].append(len(rag["add_tag"]))
        rows["addsub_start"].append(len(rag["addsub_link"]))

    def unit_target(tu):
        tag = tu.get("m_tag") or 0
        pt = tu.get("m_snapshotPoint") or {}
        kw = dict(tx=pt.get("x", np.nan) / POINT_SCALE, ty=pt.get("y", np.nan) / POINT_SCALE,
                  snap_link=tu.get("m_snapshotUnitLink", -1), snap_player=tu.get("m_snapshotControlPlayerId", -1))
        if tag:
            return dict(kw, target=TARGET_UNIT, target_tag=tag)
        return dict(kw, target=TARGET_SNAPSHOT)

    for ev in proto.decode_replay_game_events(a.read_file("replay.game.events")):
        player = slots.get(ev["_userid"]["m_userId"])
        t = ev["_event"].split(".")[-1]
        if player is None:  # observers, referees
            skipped[t] = skipped.get(t, 0) + 1
            continue
        loop = ev["_gameloop"]
        if t == "SCmdEvent":
            ab = ev.get("m_abil")
            kw = dict(flags=ev.get("m_cmdFlags", 0), sequence=ev.get("m_sequence", -1),
                      abil_link=ab["m_abilLink"] if ab else -1, abil_cmd=ab["m_abilCmdIndex"] if ab else -1)
            data = ev.get("m_data") or {}
            if "TargetPoint" in data:
                pt = data["TargetPoint"]
                kw.update(target=TARGET_POINT, tx=pt["x"] / POINT_SCALE, ty=pt["y"] / POINT_SCALE)
            elif "TargetUnit" in data:
                kw.update(unit_target(data["TargetUnit"]))
            else:
                kw.update(target=TARGET_NONE)
            row(loop, player, CMD, **kw)
        elif t == "SCmdUpdateTargetPointEvent":
            pt = ev["m_target"]
            row(loop, player, CMD_POINT, target=TARGET_POINT, tx=pt["x"] / POINT_SCALE, ty=pt["y"] / POINT_SCALE)
        elif t == "SCmdUpdateTargetUnitEvent":
            row(loop, player, CMD_UNIT, **unit_target(ev["m_target"]))
        elif t == "SCommandManagerStateEvent":
            row(loop, player, REPEAT, mgr_state=ev.get("m_state", -1), sequence=ev.get("m_sequence") or -1)
        elif t == "SSelectionDeltaEvent":
            d = ev["m_delta"]
            rm_kind, rm_val = next(iter(d["m_removeMask"].items()))
            remove = {"None": REMOVE_NONE, "Mask": REMOVE_MASK, "OneIndices": REMOVE_ONE, "ZeroIndices": REMOVE_ZERO}[rm_kind]
            row(loop, player, SELECT, group=ev["m_controlGroupId"], select_remove=remove)
            if remove == REMOVE_MASK:
                rag["rem_pos"].extend(_mask_positions(rm_val))
            elif remove in (REMOVE_ONE, REMOVE_ZERO):
                rag["rem_pos"].extend(rm_val)
            rag["add_tag"].extend(d["m_addUnitTags"])
            for sg in d["m_addSubgroups"]:
                rag["addsub_link"].append(sg["m_unitLink"])
                rag["addsub_count"].append(sg["m_count"])
        elif t == "SControlGroupUpdateEvent":
            row(loop, player, GROUP, group=ev["m_controlGroupIndex"], group_op=ev["m_controlGroupUpdate"])
        elif t == "SCameraUpdateEvent":
            tg = ev.get("m_target")
            if tg:
                row(loop, player, CAMERA, cam_x=tg["x"] / CAMERA_SCALE, cam_y=tg["y"] / CAMERA_SCALE)
        elif t == "SGameUserLeaveEvent":
            row(loop, player, LEAVE)
        elif t == "SCameraSaveEvent":
            tg = ev["m_target"]
            row(loop, player, CAMERA_SAVE, group=ev["m_which"], cam_x=tg["x"] / CAMERA_SCALE, cam_y=tg["y"] / CAMERA_SCALE)
        elif t == "STriggerPingEvent":
            pt = ev["m_point"]
            kw = dict(tx=pt["x"] / POINT_SCALE, ty=pt["y"] / POINT_SCALE,
                      flags=FLAG_MINIMAP if ev.get("m_pingedMinimap") else 0)
            if ev.get("m_unit"):
                kw.update(target=TARGET_UNIT, target_tag=ev["m_unit"], snap_link=ev.get("m_unitLink", -1))
            row(loop, player, PING, **kw)
        elif t == "SResourceTradeEvent":
            r = ev["m_resources"]
            row(loop, player, TRADE, recipient=ev["m_recipientId"], minerals=r[0], gas=r[1])
        elif t == "SResourceRequestEvent":
            r = ev["m_resources"]
            row(loop, player, REQUEST, minerals=r[0], gas=r[1])
        elif t == "SResourceRequestFulfillEvent":
            row(loop, player, REQUEST_FULFILL, request_id=ev["m_fulfillRequestId"])
        elif t == "SResourceRequestCancelEvent":
            row(loop, player, REQUEST_CANCEL, request_id=ev["m_cancelRequestId"])
        else:
            skipped[t] = skipped.get(t, 0) + 1
    out = {k: np.asarray(v, dtype=COLUMNS[k]) for k, v in rows.items()}
    out.update({k: np.asarray(v, dtype=RAGGED[k]) for k, v in rag.items()})
    out["meta"] = np.asarray(json.dumps(dict(build=build, players=players, skipped_events=skipped)))
    return out
