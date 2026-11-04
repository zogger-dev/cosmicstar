# Replay action events

Real, unedited events from `replay.game.events`, one per kind and variation, taken from a random sample of 150 1v1 replays (`worldmodel/replay_actions.py` parses them;
`worldmodel/action_tokens.py` turns them into tokens). Counts are from that sample. `_gameloop` is the event's loop, `_userid` the player's user id.

Scales: `TargetPoint` and snapshot points are map units x 4096, camera targets map units x 256. Each example's **Reads as** line is decoded from that game's own tables: ability families from the ruleset's links, unit types from its unit log, selection and groups from the tracker. Names here are labels for reading; none of them enter the model. Types are the unit's type at that moment, as the unit log records it.

## Commands

**CMD**: `SCmdEvent`. `m_abil` is null on a right-click (smart). `m_data` holds the target. Flags (`m_cmdFlags`): 0x2 queued, 0x8 smart, 0x40 set autocast, 0x80 autocast on, 0x100 user, 0x10000 minimap, 0x20000 repeat.

No target, 47997 events:

**Reads as:** Player 2, 0:12 (loop 277): NexusTrain button 0 with no target. Flags: issued by the user. Applies to the current selection.

```json
{
 "m_cmdFlags": 256,
 "m_abil": {
  "m_abilLink": 177,
  "m_abilCmdIndex": 0,
  "m_abilCmdData": null
 },
 "m_data": {
  "None": null
 },
 "m_sequence": 22,
 "m_otherUnit": null,
 "m_unitGroup": null,
 "_event": "NNet.Game.SCmdEvent",
 "_eventid": 27,
 "_gameloop": 277,
 "_userid": {
  "m_userId": 2
 },
 "_bits": 112
}
```

Point target, with an ability, 62097 events:

**Reads as:** Player 2, 0:17 (loop 392): ProtossBuild button 1 at point (102.0, 43.0). Flags: issued by the user. Applies to the current selection.

```json
{
 "m_cmdFlags": 256,
 "m_abil": {
  "m_abilLink": 172,
  "m_abilCmdIndex": 1,
  "m_abilCmdData": null
 },
 "m_data": {
  "TargetPoint": {
   "x": 417792,
   "y": 176128,
   "z": 40936
  }
 },
 "m_sequence": 47,
 "m_otherUnit": null,
 "m_unitGroup": null,
 "_event": "NNet.Game.SCmdEvent",
 "_eventid": 27,
 "_gameloop": 392,
 "_userid": {
  "m_userId": 2
 },
 "_bits": 184
}
```

Point target, right-click, 68623 events:

**Reads as:** Player 2, 0:15 (loop 350): right-click at point (101.5, 42.8). Flags: right-click, issued by the user. Applies to the current selection.

```json
{
 "m_cmdFlags": 264,
 "m_abil": null,
 "m_data": {
  "TargetPoint": {
   "x": 415757,
   "y": 175508,
   "z": 40936
  }
 },
 "m_sequence": 32,
 "m_otherUnit": null,
 "m_unitGroup": null,
 "_event": "NNet.Game.SCmdEvent",
 "_eventid": 27,
 "_gameloop": 350,
 "_userid": {
  "m_userId": 2
 },
 "_bits": 160
}
```

Unit target, with an ability, 19001 events:

**Reads as:** Player 1, 0:14 (loop 323): DroneHarvest button 0 on MineralField750 at (37.0, 144.5). Flags: issued by the user. Applies to the current selection.

```json
{
 "m_cmdFlags": 256,
 "m_abil": {
  "m_abilLink": 186,
  "m_abilCmdIndex": 0,
  "m_abilCmdData": null
 },
 "m_data": {
  "TargetUnit": {
   "m_targetUnitFlags": 111,
   "m_timer": 0,
   "m_tag": 16515073,
   "m_snapshotUnitLink": 547,
   "m_snapshotControlPlayerId": 0,
   "m_snapshotUpkeepPlayerId": 0,
   "m_snapshotPoint": {
    "x": 151552,
    "y": 591872,
    "z": 49120
   }
  }
 },
 "m_sequence": 95,
 "m_otherUnit": null,
 "m_unitGroup": null,
 "_event": "NNet.Game.SCmdEvent",
 "_eventid": 27,
 "_gameloop": 323,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 264
}
```

Unit target, right-click, 18782 events:

**Reads as:** Player 2, 0:12 (loop 291): right-click on MineralField at (140.0, 32.5). Flags: right-click, issued by the user. Applies to the current selection.

```json
{
 "m_cmdFlags": 264,
 "m_abil": null,
 "m_data": {
  "TargetUnit": {
   "m_targetUnitFlags": 111,
   "m_timer": 0,
   "m_tag": 41156609,
   "m_snapshotUnitLink": 545,
   "m_snapshotControlPlayerId": 0,
   "m_snapshotUpkeepPlayerId": 0,
   "m_snapshotPoint": {
    "x": 573440,
    "y": 133120,
    "z": 49120
   }
  }
 },
 "m_sequence": 27,
 "m_otherUnit": null,
 "m_unitGroup": null,
 "_event": "NNet.Game.SCmdEvent",
 "_eventid": 27,
 "_gameloop": 291,
 "_userid": {
  "m_userId": 2
 },
 "_bits": 240
}
```

`m_data` of kind `Data`, 82 events (the parser treats it as no target):

**Reads as:** Player 1, 4:44 (loop 6362): MedivacTransport button 3 with extra data 66846721. Flags: issued by the user, other bits 0x220. Applies to the current selection.

```json
{
 "m_cmdFlags": 800,
 "m_abil": {
  "m_abilLink": 140,
  "m_abilCmdIndex": 3,
  "m_abilCmdData": null
 },
 "m_data": {
  "Data": 66846721
 },
 "m_sequence": 527,
 "m_otherUnit": null,
 "m_unitGroup": null,
 "_event": "NNet.Game.SCmdEvent",
 "_eventid": 27,
 "_gameloop": 6362,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 144
}
```

**CMD_POINT**: `SCmdUpdateTargetPointEvent`, re-target the last command to a point, 176224 events:

**Reads as:** Player 1, 0:11 (loop 248): changes the last command's target to point (68.3, 130.0).

```json
{
 "m_target": {
  "x": 279729,
  "y": 532413,
  "z": 40936
 },
 "_event": "NNet.Game.SCmdUpdateTargetPointEvent",
 "_eventid": 104,
 "_gameloop": 248,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 96
}
```

**CMD_UNIT**: `SCmdUpdateTargetUnitEvent`, re-target it to a unit, 59413 events:

**Reads as:** Player 1, 0:09 (loop 210): changes the last command's target to VespeneGeyser at (35.5, 136.5).

```json
{
 "m_target": {
  "m_targetUnitFlags": 111,
  "m_timer": 0,
  "m_tag": 21757953,
  "m_snapshotUnitLink": 550,
  "m_snapshotControlPlayerId": 0,
  "m_snapshotUpkeepPlayerId": 0,
  "m_snapshotPoint": {
   "x": 145408,
   "y": 559104,
   "z": 49120
  }
 },
 "_event": "NNet.Game.SCmdUpdateTargetUnitEvent",
 "_eventid": 105,
 "_gameloop": 210,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 176
}
```

**REPEAT**: `SCommandManagerStateEvent`, repeat the last command, 313647 events:

**Reads as:** Player 1, 0:09 (loop 210): repeats the last command (state 1, sequence 88).

```json
{
 "m_state": 1,
 "m_sequence": 88,
 "_event": "NNet.Game.SCommandManagerStateEvent",
 "_eventid": 103,
 "_gameloop": 210,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 56
}
```

CMD_POINT, CMD_UNIT and REPEAT become one REISSUE token (held hotkeys fold into a press count).

## Selection

**SELECT**: `SSelectionDeltaEvent`. `m_controlGroupId` 10 is the active selection, 0-9 a control group. It holds no screen position.

`m_removeMask` says which current members leave, by position in that list before the change:

| Value | Meaning |
|---|---|
| `None` | Nobody leaves |
| `Mask` | A bitmask of positions to remove |
| `OneIndices` | The positions to remove |
| `ZeroIndices` | The positions to keep; the rest are removed |

Active selection, keep positions, 187362 events:

**Reads as:** Player 2, 0:09 (loop 205): changes the active selection: keeps only positions [], removing the rest, adds 11 Probe. Before: Nexus. After this loop: Nexus. Another selection or group event falls on this same loop, so that after-state reflects both.

```json
{
 "m_controlGroupId": 10,
 "m_delta": {
  "m_subgroupIndex": 0,
  "m_removeMask": {
   "ZeroIndices": []
  },
  "m_addSubgroups": [
   {
    "m_unitLink": 107,
    "m_subgroupPriority": 33,
    "m_intraSubgroupPriority": 1,
    "m_count": 11
   }
  ],
  "m_addUnitTags": [
   62128129,
   62390273,
   62652417,
   62914561,
   63176705,
   63700993,
   63963137,
   64225281,
   64487425,
   64749569,
   65011713
  ]
 },
 "_event": "NNet.Game.SSelectionDeltaEvent",
 "_eventid": 28,
 "_gameloop": 205,
 "_userid": {
  "m_userId": 2
 },
 "_bits": 456
}
```

Active selection, bitmask, 44691 events:

**Reads as:** Player 1, 0:17 (loop 395): changes the active selection: removes positions [0] (bitmask), adds Egg. Before: Egg, Larva. After this loop: Egg, Larva.

```json
{
 "m_controlGroupId": 10,
 "m_delta": {
  "m_subgroupIndex": 0,
  "m_removeMask": {
   "Mask": [
    1,
    1
   ]
  },
  "m_addSubgroups": [
   {
    "m_unitLink": 126,
    "m_subgroupPriority": 54,
    "m_intraSubgroupPriority": 1,
    "m_count": 1
   }
  ],
  "m_addUnitTags": [
   58195969
  ]
 },
 "_event": "NNet.Game.SSelectionDeltaEvent",
 "_eventid": 28,
 "_gameloop": 395,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 136
}
```

Active selection, remove positions, 10902 events:

**Reads as:** Player 1, 1:23 (loop 1880): changes the active selection: removes positions [8], adds nothing. Before: 11 Drone. After this loop: 10 Drone.

```json
{
 "m_controlGroupId": 10,
 "m_delta": {
  "m_subgroupIndex": 0,
  "m_removeMask": {
   "OneIndices": [
    8
   ]
  },
  "m_addSubgroups": [],
  "m_addUnitTags": []
 },
 "_event": "NNet.Game.SSelectionDeltaEvent",
 "_eventid": 28,
 "_gameloop": 1880,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 72
}
```

Active selection, nobody removed, 23234 events:

**Reads as:** Player 1, 0:45 (loop 1012): changes the active selection: removes nobody, adds 2 Egg. Before: Egg. After this loop: 3 Egg.

```json
{
 "m_controlGroupId": 10,
 "m_delta": {
  "m_subgroupIndex": 0,
  "m_removeMask": {
   "None": null
  },
  "m_addSubgroups": [
   {
    "m_unitLink": 126,
    "m_subgroupPriority": 54,
    "m_intraSubgroupPriority": 1,
    "m_count": 2
   }
  ],
  "m_addUnitTags": [
   66584577,
   67633153
  ]
 },
 "_event": "NNet.Game.SSelectionDeltaEvent",
 "_eventid": 28,
 "_gameloop": 1012,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 160
}
```

Control group, keep positions, 1271 events:

**Reads as:** Player 1, 0:35 (loop 793): changes control group 0: keeps only positions [], removing the rest, adds Overlord. Before: nothing. After this loop: Overlord.

```json
{
 "m_controlGroupId": 0,
 "m_delta": {
  "m_subgroupIndex": 0,
  "m_removeMask": {
   "ZeroIndices": []
  },
  "m_addSubgroups": [
   {
    "m_unitLink": 129,
    "m_subgroupPriority": 72,
    "m_intraSubgroupPriority": 1,
    "m_count": 1
   }
  ],
  "m_addUnitTags": [
   57933827
  ]
 },
 "_event": "NNet.Game.SSelectionDeltaEvent",
 "_eventid": 28,
 "_gameloop": 793,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 136
}
```

Control group, bitmask, 6654 events:

**Reads as:** Player 1, 2:01 (loop 2724): changes control group 1: removes positions [0] (bitmask), adds 2 Zergling. Before: Egg. After this loop: 2 Zergling, Egg.

```json
{
 "m_controlGroupId": 1,
 "m_delta": {
  "m_subgroupIndex": 1,
  "m_removeMask": {
   "Mask": [
    1,
    1
   ]
  },
  "m_addSubgroups": [
   {
    "m_unitLink": 128,
    "m_subgroupPriority": 68,
    "m_intraSubgroupPriority": 1,
    "m_count": 2
   }
  ],
  "m_addUnitTags": [
   73400321,
   73662465
  ]
 },
 "_event": "NNet.Game.SSelectionDeltaEvent",
 "_eventid": 28,
 "_gameloop": 2724,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 168
}
```

Control group, remove positions, 6580 events:

**Reads as:** Player 1, 3:07 (loop 4198): changes control group 1: removes positions [10], adds 2 Zergling. Before: 12 Zergling, Egg. After this loop: 14 Zergling, Egg. Another selection or group event falls on this same loop, so that after-state reflects both.

```json
{
 "m_controlGroupId": 1,
 "m_delta": {
  "m_subgroupIndex": 0,
  "m_removeMask": {
   "OneIndices": [
    10
   ]
  },
  "m_addSubgroups": [
   {
    "m_unitLink": 128,
    "m_subgroupPriority": 68,
    "m_intraSubgroupPriority": 1,
    "m_count": 2
   }
  ],
  "m_addUnitTags": [
   78381061,
   81002497
  ]
 },
 "_event": "NNet.Game.SSelectionDeltaEvent",
 "_eventid": 28,
 "_gameloop": 4198,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 176
}
```

Control group, nobody removed, 1 events:

**Reads as:** Player 1, 14:48 (loop 19896): changes control group 7: removes nobody, adds ChangelingZealot. Before: Overseer. After this loop: Overseer, ChangelingZealot. Another selection or group event falls on this same loop, so that after-state reflects both.

```json
{
 "m_controlGroupId": 7,
 "m_delta": {
  "m_subgroupIndex": 0,
  "m_removeMask": {
   "None": null
  },
  "m_addSubgroups": [
   {
    "m_unitLink": 32,
    "m_subgroupPriority": 64,
    "m_intraSubgroupPriority": 1,
    "m_count": 1
   }
  ],
  "m_addUnitTags": [
   111673358
  ]
 },
 "_event": "NNet.Game.SSelectionDeltaEvent",
 "_eventid": 28,
 "_gameloop": 19896,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 128
}
```

**What the added units are** (60 games, both players, events with no other selection or group event on that loop). State is the tracker's selection and group bits just before and after the event.

| Check | Active selection (54,495 events with adds) | Control groups (2,833 events with adds) |
|---|---|---|
| Adds are exactly the selection after | 43,088 (79%) | 1 |
| Adds are exactly the selection before | 1,375 (2.5%) | 1 |
| Adds share no unit with the selection before | 47,882 (88%) | 2,830 (99.9%) |
| Adds are in that control group after | n/a | 2,810 (99%) |

Some of these events are the game's doing, not a decision: a larva morphing shows as `Egg` replacing it in the selection, and hatching eggs as `Zergling` joining a control group, with no command from the player at that moment.

So an active-selection event usually sets what the selection becomes, not what it was. In control groups, 84% of added units (4,848 of 5,801) first appear within 4 loops: mostly new units joining a group, such as units leaving eggs.

**GROUP**: `SControlGroupUpdateEvent`. `m_controlGroupIndex` is the group; `m_controlGroupUpdate`: 0 set, 1 add, 2 recall, 3 clear, 4 set and steal, 5 add and steal. `m_mask` has the same four values. On a recall it lists group members left out of the selection, for example inside a transport; on set and add it is positions of the group's current list.

Set, mask `None`, 14658 events:

**Reads as:** Player 1, 0:18 (loop 408): set control group 0; mask removes nobody. Group before: nothing. After this loop: Egg.

```json
{
 "m_controlGroupIndex": 0,
 "m_controlGroupUpdate": 0,
 "m_mask": {
  "None": null
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 408,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 32
}
```

Add, mask `None`, 5402 events:

**Reads as:** Player 1, 1:18 (loop 1752): add to control group 4; mask removes nobody. Group before: Hatchery. After this loop: 2 Hatchery.

```json
{
 "m_controlGroupIndex": 4,
 "m_controlGroupUpdate": 1,
 "m_mask": {
  "None": null
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 1752,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 32
}
```

Add, mask `Mask`, 1094 events:

**Reads as:** Player 1, 4:56 (loop 6649): add to control group 1; mask removes positions [1, 8] (bitmask). Group before: 12 Zergling. After this loop: 13 Zergling.

```json
{
 "m_controlGroupIndex": 1,
 "m_controlGroupUpdate": 1,
 "m_mask": {
  "Mask": [
   9,
   5
  ]
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 6649,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 48
}
```

Add, mask `OneIndices`, 678 events:

**Reads as:** Player 1, 9:45 (loop 13119): add to control group 1; mask removes positions [4, 36, 51, 57, 61]. Group before: 55 Zergling, 18 Roach, 6 Ravager. After this loop: 54 Zergling, 18 Roach, 15 Larva, 6 Ravager, Egg.

```json
{
 "m_controlGroupIndex": 1,
 "m_controlGroupUpdate": 1,
 "m_mask": {
  "OneIndices": [
   4,
   36,
   51,
   57,
   61
  ]
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 13119,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 88
}
```

Add, mask `ZeroIndices`, 13 events:

**Reads as:** Player 1, 18:42 (loop 25148): add to control group 1; mask keeps only positions [], removing the rest. Group before: nothing. After this loop: 16 Egg.

```json
{
 "m_controlGroupIndex": 1,
 "m_controlGroupUpdate": 1,
 "m_mask": {
  "ZeroIndices": []
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 25148,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 40
}
```

Recall, mask `None`, 415587 events:

**Reads as:** Player 1, 0:09 (loop 202): recall control group 1; mask removes nobody. Group before: Overlord. After this loop: Overlord.

```json
{
 "m_controlGroupIndex": 1,
 "m_controlGroupUpdate": 2,
 "m_mask": {
  "None": null
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 202,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 32
}
```

Recall, mask `Mask`, 21894 events:

**Reads as:** Player 1, 4:55 (loop 6622): recall control group 1; mask removes positions [1] (bitmask). Group before: 13 Zergling. After this loop: 13 Zergling.

```json
{
 "m_controlGroupIndex": 1,
 "m_controlGroupUpdate": 2,
 "m_mask": {
  "Mask": [
   2,
   2
  ]
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 6622,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 40
}
```

Recall, mask `OneIndices`, 7372 events:

**Reads as:** Player 1, 9:43 (loop 13079): recall control group 1; mask removes positions [57]. Group before: 50 Zergling, 18 Roach, 6 Ravager, 3 Egg. After this loop: 50 Zergling, 18 Roach, 6 Ravager, 3 Egg.

```json
{
 "m_controlGroupIndex": 1,
 "m_controlGroupUpdate": 2,
 "m_mask": {
  "OneIndices": [
   57
  ]
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 13079,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 48
}
```

Recall, mask `ZeroIndices`, 123 events:

**Reads as:** Player 2, 11:20 (loop 15246): recall control group 4; mask keeps only positions [7, 14], removing the rest. Group before: 2 Zergling. After this loop: 2 Zergling.

```json
{
 "m_controlGroupIndex": 4,
 "m_controlGroupUpdate": 2,
 "m_mask": {
  "ZeroIndices": [
   7,
   14
  ]
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 15246,
 "_userid": {
  "m_userId": 5
 },
 "_bits": 56
}
```

Clear, mask `None`, 1018 events:

**Reads as:** Player 1, 5:07 (loop 6878): clear control group 3; mask removes nobody. Group before: Zergling. After this loop: nothing.

```json
{
 "m_controlGroupIndex": 3,
 "m_controlGroupUpdate": 3,
 "m_mask": {
  "None": null
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 6878,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 32
}
```

Set and steal, mask `None`, 6919 events:

**Reads as:** Player 1, 3:30 (loop 4725): set and steal into control group 2; mask removes nobody. Group before: 2 Zergling. After this loop: Zergling.

```json
{
 "m_controlGroupIndex": 2,
 "m_controlGroupUpdate": 4,
 "m_mask": {
  "None": null
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 4725,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 32
}
```

Add and steal, mask `None`, 1753 events:

**Reads as:** Player 1, 13:20 (loop 17930): add and steal into control group 3; mask removes nobody. Group before: Roach. After this loop: 9 Roach, 3 Zergling, Drone.

```json
{
 "m_controlGroupIndex": 3,
 "m_controlGroupUpdate": 5,
 "m_mask": {
  "None": null
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 17930,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 32
}
```

Add and steal, mask `Mask`, 140 events:

**Reads as:** Player 1, 12:40 (loop 17026): add and steal into control group 3; mask removes positions [0, 7, 10] (bitmask). Group before: 7 Zergling, Ravager, Roach, Egg. After this loop: 15 Roach, 8 Zergling, Ravager, Egg.

```json
{
 "m_controlGroupIndex": 3,
 "m_controlGroupUpdate": 5,
 "m_mask": {
  "Mask": [
   11,
   1036
  ]
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 17026,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 56
}
```

Add and steal, mask `OneIndices`, 76 events:

**Reads as:** Player 1, 6:42 (loop 9010): add and steal into control group 2; mask removes positions [13]. Group before: 7 Stalker, 7 Probe. After this loop: 7 Stalker, 7 Probe.

```json
{
 "m_controlGroupIndex": 2,
 "m_controlGroupUpdate": 5,
 "m_mask": {
  "OneIndices": [
   13
  ]
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 9010,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 48
}
```

Add and steal, mask `ZeroIndices`, 2 events:

**Reads as:** Player 2, 12:09 (loop 16337): add and steal into control group 3; mask keeps only positions [10, 15, 53, 55, 61], removing the rest. Group before: 5 Zergling, 3 LurkerMPBurrowed, Overseer. After this loop: 40 Zergling.

```json
{
 "m_controlGroupIndex": 3,
 "m_controlGroupUpdate": 5,
 "m_mask": {
  "ZeroIndices": [
   10,
   15,
   53,
   55,
   61
  ]
 },
 "_event": "NNet.Game.SControlGroupUpdateEvent",
 "_eventid": 29,
 "_gameloop": 16337,
 "_userid": {
  "m_userId": 1
 },
 "_bits": 88
}
```

## Camera

**CAMERA**: `SCameraUpdateEvent`, 1987430 events. Of 198,000 sampled from other games, non-empty: `m_target` 92%, `m_reason` 91%, `m_distance` 19%, `m_follow` 11%, `m_pitch` 3.5%, `m_yaw` 1.7%. Only `m_target` is used, and events without one are dropped.

**Reads as:** Player 2, 0:09 (loop 202): moves the camera to (133.5, 30.8).

```json
{
 "m_target": {
  "x": 34176,
  "y": 7874
 },
 "m_distance": null,
 "m_pitch": null,
 "m_yaw": null,
 "m_reason": 5,
 "m_follow": false,
 "_event": "NNet.Game.SCameraUpdateEvent",
 "_eventid": 49,
 "_gameloop": 202,
 "_userid": {
  "m_userId": 2
 },
 "_bits": 72
}
```

**CAMERA_SAVE**: `SCameraSaveEvent`, 6903 events:

**Reads as:** Player 1, 0:09 (loop 220): saves camera hotkey 0 at (42.5, 137.8).

```json
{
 "m_which": 0,
 "m_target": {
  "x": 10880,
  "y": 35266
 },
 "_event": "NNet.Game.SCameraSaveEvent",
 "_eventid": 14,
 "_gameloop": 220,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 56
}
```

## Others

**PING**: `STriggerPingEvent`, 6 events:

**Reads as:** Player 1, 9:17 (loop 12482): pings (44.5, 143.3).

```json
{
 "m_point": {
  "x": 182468,
  "y": 586780
 },
 "m_unit": 0,
 "m_unitLink": 65535,
 "m_unitControlPlayerId": null,
 "m_unitUpkeepPlayerId": null,
 "m_unitPosition": {
  "x": 0,
  "y": 0,
  "z": 0
 },
 "m_unitIsUnderConstruction": false,
 "m_pingedMinimap": false,
 "m_option": -1,
 "_event": "NNet.Game.STriggerPingEvent",
 "_eventid": 36,
 "_gameloop": 12482,
 "_userid": {
  "m_userId": 3
 },
 "_bits": 240
}
```

**LEAVE**: `SGameUserLeaveEvent`, 522 events:

**Reads as:** Player 1, 18:53 (loop 25385): leaves the game.

```json
{
 "m_leaveReason": 0,
 "_event": "NNet.Game.SGameUserLeaveEvent",
 "_eventid": 101,
 "_gameloop": 25385,
 "_userid": {
  "m_userId": 0
 },
 "_bits": 24
}
```

**TRADE, REQUEST, REQUEST_FULFILL, REQUEST_CANCEL**: `SResourceTradeEvent`, `SResourceRequestEvent`, `SResourceRequestFulfillEvent`, `SResourceRequestCancelEvent`. No examples: none occur in this 1v1 sample. They need a teammate.

## Seen in replays but not made into tokens

`STriggerDialogControlEvent` (406), `SUnitClickEvent` (2, a tag only, no coordinates), and sync and options events.

**Reads as:** Player 2, 15:46 (loop 21196): clicks Artosilope.

```json
{
 "m_unitTag": 27525121,
 "_event": "NNet.Game.SUnitClickEvent",
 "_eventid": 39,
 "_gameloop": 21196,
 "_userid": {
  "m_userId": 1
 },
 "_bits": 56
}
```
