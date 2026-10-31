# CosmicStar

A learned simulator of StarCraft II, trained on replays, as the first stage of an
AlphaStar-like agent. The engine plays replays back and the full game state is extracted
loop by loop; a model learns to predict every entity's next state from the present and the
players' commands. Design notes, measurements and the record of what did and did not work:
[`src/cosmicstar/worldmodel/README.md`](src/cosmicstar/worldmodel/README.md).

## Layout

| | |
|---|---|
| `scripts/extract_states.py` | plays replays back in the engine (observer view, and each player's view with fog) and dumps the state every N loops: units, orders, production, buffs, deaths, visibility, the player's selection and command recipients |
| `scripts/extract_actions.py` | every player's interface actions from the replay file (no engine): commands, selections, control groups, camera |
| `scripts/dump_rules.py`, `scripts/export_balance.py`, `scripts/build_link_table.py` | the game's static data per ruleset (unit and ability catalogs, from the engine and the Editor's balance export), and the replay's ability numbering mapped to names |
| `scripts/extract_gamedata.py`, `scripts/build_casclib.sh` | the game's own data catalogs (GameData XMLs of the ladder mods) read out of the CASC storage through CascLib, built once with the shell script |
| `scripts/build_ruleset_tables.py` | the balance exports, rules dumps and game-data upgrade effects as per-ruleset graphs of unit, ability and upgrade nodes with fixed-width features, the input of the ruleset encoder |
| `scripts/dump_unit_data.py`, `scripts/make_splits.py`, `scripts/download_maps.py`, `scripts/check_player_commands.py` | unit metadata, the train / held-out split, maps from Blizzard's depot, validation of the tracked selections against the engine |
| `scripts/train_direct.py` | the direct state-space simulator: state at t plus the commands in [t, t+k) to every entity's state at t+k |
| `scripts/regen_s2proto.py` | rebuilds the protobuf bindings for the fast parser and patches s2protocol for Python 3.12+ |
| `src/cosmicstar/worldmodel/` | tokenizers for states and actions, the selection tracker and command join, the ruleset encoder (types from the balance data), the encoders, the latent, the direct model |
| `src/cosmicstar/envs/` | running the Windows client under Wine/Proton (`wine_run_config.py`), maps and mods from the depot (`depot.py`), older game builds from Blizzard's CDN (`tact.py`), files out of the CASC storage (`casc.py`, needs `lib/libcasc.so`) |

`data/` (gitignored) holds replays, extractions, balance exports and training runs.

## Setup

Python 3.12 in a virtual environment (`uv venv --python 3.12 .venv`, then
`uv pip install -e . --all-extras`), then `python scripts/regen_s2proto.py` once so the
protobuf bindings load with the fast parser. `requirements.txt` is the lock
(`uv pip compile pyproject.toml --all-extras -o requirements.txt`).

## Running on Linux (Wine/Proton)

Blizzard's headless Linux builds stop at 4.10; every 5.0.x client is Windows only. On Linux the
client runs inside a Wine prefix: a Lutris Battle.net install under GE-Proton, launched through
`umu-run`. `src/cosmicstar/envs/wine_run_config.py` registers a pysc2 run config that finds the
install (`$SC2PATH`, else `~/Games/battlenet/...`), rewrites paths for the client and sets the
Proton environment, so `scripts/extract_states.py` and the other engine-driven scripts run
unchanged. The client is windowed (no headless mode) and needs a display.

A replay only plays in the engine build it was recorded on. `python -m cosmicstar.envs.tact
<build>` fetches an older build from Blizzard's CDN into a side copy of the install
("StarCraft II old"), and the run config picks the install per build. The Editor's balance
export (`scripts/export_balance.py`) runs under the same prefix.

## Running on Windows

The scripts find the standard install (`C:\Program Files (x86)\StarCraft II`); older builds
are downloaded by the Battle.net client when a replay of that version is opened.

## Dependencies

`pysc2`, `s2protocol`, `mpyq`, `numpy`, `torch`, `requests`, `tqdm`, `absl-py`; `sc2reader`
only for map dependency scanning in `download_maps.py`; `matplotlib` for analysis plots.
