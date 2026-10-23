# Build-order files (`.bo`)

`scripts/zerg_bench.py --build <name|path>` plays one of these in a real game.
Lines run **in order**; each is `[&] <trigger> <action> [args]`. `#` starts a comment.

## Triggers

| Trigger | Waits until |
|---|---|
| `N` | supply (incl. eggs) reaches N. Drones are made up to N and **no further** while this line is pending, so `16 hatch natural` means "drone to 16, then hatch". |
| `@pool`, `@natural`, `@third`, `@main` | that building has finished |
| `@queen:<base>` | the first queen trained at that base's hatchery has popped |
| `-` (or nothing) | nothing: runs straight after the previous line |

Once a trigger has fired it stays fired (the extractor trick drops supply below its trigger, for example).

A leading `&` makes the line a **background** step: it is registered when reached and then runs whenever its trigger and resources allow, without holding up later lines.

While a step's trigger has fired but it can't be afforded yet, minerals are saved for it (no drones are made).

## Actions

| Action | |
|---|---|
| `overlord [main\|natural\|third]` | from that hatchery's larva (default main) |
| `hatch natural\|third` | a mining drone is pulled on a timer so it arrives as the bank reaches the hatch cost; done when the hatch starts |
| `extractor [main\|natural]` | builder: the next `rally extractor` drone if one is queued, else the closest mining drone; placed when the trigger fires |
| `pool` | builder as for `extractor`; placed the moment we can afford it |
| `extractor_trick` | extractor, then a drone, then cancel the extractor (drone comes back: +1 supply over cap) |
| `queen main\|natural\|third` | from that base's hatchery |
| `lings N` | N pairs of zerglings |
| `ling_speed` | metabolic boost, as soon as the pool is done and we have the gas |
| `rally <role> [xN]` | the next N drones made go to: `main`, `natural`, `third`, `gas`, `extractor` (builder), `pool` (builder) |
| `default main\|natural\|third\|fill` | where drones without a queued role go; `fill` = main, then natural, then third, 16 mineral drones each |
| `pull_gas N` | stop mining gas once collected + in-flight gas reaches N; gas drones go to the default |
| `auto_overlords` | from here on, make overlords when supply is about to run out |
| `include <file>` | splice in another build file (path relative to this one) |

Standing behaviour (not scripted): pro split at the start, main-base drones held on their patch (no bouncing),
natural/third drones rallied to one patch and left to spread out, injects on cooldown with one queen per hatchery.
