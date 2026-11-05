# Retrieval, Structure, and Ingestion of the Blizzard StarCraft II AI Replay Corpus

Saved 2026-11-04 from a research write-up the user pasted. None of it has been tried here yet. Its only claim checked against our files: Pack 1 is 64,396 entries in our `data/downloads/blizzard-3.16.1-pack1/list.txt`, where the text below says 64,327 parsed matches.

## Checked 2026-11-04, no key and no downloads

- Still served: Pack 1 fix 2.61 GB, Pack 2 47.44 GB, Linux 3.16.1 2.82 GB, 4.0.2 3.46 GB and 4.10 4.12 GB (HTTP 200 to a HEAD request). `s2client-proto` still lists both packs and has `download_replays.py`, last changed 2022-01-14.
- Pack 2 is 47.44 GB in decimal units, which fits the 44.2 GB below if that is GiB. Its game count was not checked.
- The replay API works with a developer key: token from `https://oauth.battle.net/token`, namespace `s2-client-replays`, packs served from `http://blzdistsc2-a.akamaihd.net/ReplayPacks/`. The script in `s2client-proto` uses `us.battle.net/oauth/token`, which refused a request here. Credentials are in `~/.config/cosmicstar/blizzard.env`, outside the repo.
- `s2client-proto/buildinfo/versions.json` stops at 4.10.1, but the API goes further: 4.11.0-4.11.2, 5.0.2 and 5.0.4. 5.0.0, 5.0.1, 5.0.3 and 5.0.5-5.0.15 returned nothing, so nothing near our 5.0.14 and 5.0.15.
- Every pack counted from the last 22 bytes of its zip, no replay downloaded (`data/downloads/blizzard-api/manifest.json`): 6,008 packs, of which 3,240 are empty, holding 24.8M files in 912 GB. These are files, assumed to be one replay each; nothing has been opened. Client version 5.0.2 (build 81433): 5,443,573 files, 198.6 GB. 5.0.4 (build 82457): 2,078,611 files, 75.6 GB.
- Not known yet: game format and MMR mix, which maps, and whether those builds run here.

## Architecture and Scope of the StarCraft II Replay Corpus

The StarCraft II machine learning corpus originated from a joint research initiative between Blizzard Entertainment and Google DeepMind designed to establish real-time strategy environments as grand challenges for reinforcement learning. The initial public release of the StarCraft II Learning Environment in August 2017 provided an immediate public baseline of approximately 65,000 anonymized matches, accompanied by commitments to scale the corpus beyond 500,000 to over one million games. The perception that Blizzard published only a single package of roughly 60,000 games stems from the distribution format of those initial releases: monolithic zip files hosted directly on static content delivery network (CDN) endpoints.

The primary static archive, designated as Patch 3.16.1 Pack 1, contains exactly 64,327 parsed matches and serves as the standard lightweight test bed for local environment validation. To deliver larger volumes without causing storage bottlenecks, Blizzard released a secondary static archive, Patch 3.16.1 Pack 2, which added 44.2 GB of compressed match logs, encompassing several hundred thousand additional games.

Beyond static file hosting, Blizzard migrated the distribution of the broader dataset to an automated, programmatic infrastructure powered by the Blizzard Game Data APIs and the s2client-proto interface. As documented in offline reinforcement learning benchmarks such as AlphaStar Unplugged, Blizzard has indexed and exposed approximately 20 million anonymized human ladder games across multiple client iterations. In empirical research, restricting the data to stable competitive windows, such as client versions 4.8.2 through 4.9.2, yields an operational pool of nearly 5 million matches. Applying competitive skill thresholds, specifically isolating players with Matchmaking Ratings (MMR) exceeding 3,500 (representing the top 22% of human ladder competitors), filters this corpus down to approximately 1.4 million high-tier matches. Because StarCraft II is a two-player adversarial domain, each game supplies two discrete trajectories, resulting in 2.8 million episodes representing more than 30 years of cumulative human strategic play.

| Dataset tier | Access modality | Client versions | Approximate match count | File footprint | Intended computational use |
|---|---|---|---|---|---|
| Static Pack 1 | Direct HTTP / CDN | 3.16.1 | 64,327 replays | 2.61 GB zip / 7.5 GB uncompressed | Environment verification and pipeline sanity testing |
| Static Pack 2 | Direct HTTP / CDN | 3.16.1 | ~400,000 replays | 44.2 GB zip | Single-version supervised imitation learning |
| Blizzard Replay API corpus | OAuth 2.0 batch retrieval script | 3.16.1 to 5.0.x | ~20,000,000 replays | Multi-terabyte scale | Large-scale offline reinforcement learning and sequence modeling |
| AlphaStar Unplugged subset | Replay API subsetting (MMR > 3500) | 4.8.2 to 4.9.2 | 1,400,000 replays (2.8M episodes) | Tens of terabytes | Standardized offline reinforcement learning benchmarks |

## Direct Static Download Repositories and File System Architecture

Researchers seeking immediate local corpora without establishing cloud API credentials can utilize Blizzard's persistent static archives hosted on its Akamai CDN infrastructure. Access to these archives is subject to the StarCraft II AI and Machine Learning License Agreement, which is legally enforced through archive encryption.

### Direct static download endpoints

The static archives encompass both replay packages and the precise map caches required to parse them offline. The persistent URLs include:

- Static Replay Pack 1 (corrected release): http://blzdistsc2-a.akamaihd.net/ReplayPacks/3.16.1-Pack_1-fix.zip
  Note: the initial 3.16.1-Pack_1.zip upload lacked critical Battle.net cache artifacts required for headless simulation; the -fix package contains the complete file tree.
- Static Replay Pack 2: http://blzdistsc2-a.akamaihd.net/ReplayPacks/3.16.1-Pack_2.zip
- Standard Melee Map Pack: http://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip
- Competitive Ladder Season Map Packs:
  - http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season1.zip
  - http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season2.zip
  - http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season3.zip
  - http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season4.zip
  - http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2018Season2_Updated.zip
  - http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2019Season1.zip
- Headless Linux Game Binaries:
  - http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip
  - http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.0.2.zip
  - http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip

### Archive decryption and installation topology

All compressed archives distributed by Blizzard for artificial intelligence research, including headless Linux binaries, map pools, and replay packs, are secured with the standard extraction password: `iagreetotheeula`. Entering this decryption key represents programmatic acceptance of the licensing terms governing non-commercial research use.

Replay parsing cannot execute in isolation from the game assets. When unpacking replay archives, the resulting directories cannot simply be placed into an arbitrary workspace. StarCraft II relies on internal hash tables to cross-reference game assets, unit definitions, and map terrain. Extracted files must be merged directly into the core StarCraft II installation hierarchy. The default root paths are:

- Windows: `C:\Program Files (x86)\StarCraft II\`
- macOS: `/Applications/StarCraft II/`
- Linux headless: the base extraction directory of the headless Linux build (for example, `/home/user/StarCraftII/`)

Within the root directory, the extracted Battle.net cache folder and Replays folder must overwrite or combine with the corresponding subdirectories. The installation requires a specific topology: a `Battle.net/Cache` subdirectory containing map cache hashes, a `Maps` directory holding season map files, a `Replays` directory containing raw `.SC2Replay` files, and `SC2Data` alongside `Versions` housing executable build libraries. If the Battle.net cache is absent, the client engine aborts when attempting to deserialize map dependencies during replay initialization.

## Accessing the Multi-Million Replay Corpus via the Blizzard Game Data API

To retrieve the full dataset of one million to twenty million games, automated extraction through Blizzard's Game Data API is required. Moving away from static archive hosting resolved petabyte-scale storage bottlenecks for Blizzard and allowed dynamic querying across ladder matchmaking cycles.

The retrieval pipeline operates by having an automated Python client query Blizzard's Battle.net OAuth services, retrieve signed batch manifest URLs partitioned by client version hash, and sequentially download and unpack the multi-part replay archives directly into local storage pools.

### Authentication and developer registration

Interfacing with the replay API requires registered developer credentials issued by Blizzard:

1. The researcher creates an account on the Blizzard Developer Portal (https://develop.battle.net).
2. Under the API Access and Client Management console, a new application client is generated.
3. The platform provides an OAuth Client ID (API Key) and Client Secret, which grant access to StarCraft II game data endpoints.

### Ingestion tooling: download_replays.py

Blizzard provides an official extraction utility within the Blizzard/s2client-proto repository located under `samples/replay-api/download_replays.py`. The utility interfaces directly with Blizzard's version manifests, which map client version strings to internal build hashes through `/buildinfo/versions.json`.

The script dependencies are installed via Python's package manager:

```bash
git clone https://github.com/Blizzard/s2client-proto.git
cd s2client-proto/samples/replay-api
pip install -r requirements.txt
```

Execution requires specifying the OAuth parameters, the target game version, and the target storage path:

```bash
python download_replays.py \
  --key <BLIZZARD_CLIENT_KEY> \
  --secret <BLIZZARD_CLIENT_SECRET> \
  --version 4.10.0 \
  --replays_dir /data/starcraft_replays/4.10.0/ \
  --extract
```

The `--version` argument ensures that all downloaded matches align with a single deterministic client build. The `--extract` flag decompresses the streamed multi-part archives automatically using the standard EULA password. For researchers targeting specific subsets, the script supports a `--replayset_csv` flag, which ingests a list of designated replay file hashes to download only pre-selected, quality-filtered matches.

During long-running ingestion jobs across millions of files, researchers must account for API rate limits and token expirations. When querying older or transitional patch manifests, certain replay batches may return empty zip payloads due to archival pruning. Pipeline scripts should validate the byte size of each downloaded bundle, refresh expired OAuth bearer tokens, and focus retrieval on major stable releases, such as versions 4.8.2 through 4.10.0, where replay availability remains dense.

## Technical Ingestion, Parsing Paradigms, and Skill Normalization

An `.SC2Replay` file does not contain pre-rendered video frames or comprehensive spatial state logs. Instead, StarCraft II replays function as dense, deterministic event journals containing initial random seeds, player configuration metadata, and the timestamped action commands executed by each participant.

### Deterministic engine execution and version locking

Extracting ground-truth feature representations requires the StarCraft II engine to re-simulate the entire match step-by-step from the recorded action stream. This design imposes a strict version-locking requirement: a replay recorded on patch 4.8.2 cannot be accurately parsed by a patch 4.10.0 executable. Any variance in balance data, unit velocities, weapon cooldowns, or collision geometry causes the simulation to diverge from the historical match, resulting in desynchronization.

In distributed machine learning environments, researchers resolve this by deploying containerized Linux workers, pairing version-pinned headless binaries (such as SC2.4.10.zip) with corresponding replay archives. The headless binaries execute without graphics hardware, advancing the game clock as quickly as CPU cores can compute state updates.

### Comparative analysis of replay parsing frameworks

Depending on whether an agent requires dense spatial observations or abstract discrete token streams, researchers select between three primary parsing interfaces.

| Parsing interface | Underlying mechanism | Processing velocity | Extracted modalities | Primary research application |
|---|---|---|---|---|
| PySC2 (DeepMind) | StarCraft II client protobuf API | Low (~10–50 steps/second per core) | Multi-channel spatial feature planes, screen/minimap tensors, unit health, visibility masks | Spatial reinforcement learning, end-to-end vision-based agents |
| s2protocol (Blizzard) | Low-level bitstream parsing of the internal MPQ archive | Extremely high (>100x speedup over PySC2) | Game events, discrete player command IDs, unit coordinate clicks, timestamped actions | Autoregressive transformer modeling, build-order forecasting, high-throughput behavioral cloning |
| sc2reader (community) | Python object abstraction built over s2protocol | Moderate (~5–10x speedup over PySC2) | Pythonic object graphs (Player, Message, Unit, Ability, Selection) | Dataset indexing, metadata extraction, player MMR verification, game outcome labeling |

For investigations relying on spatial observation spaces, PySC2 is essential because it captures dynamic visibility, terrain height maps, and unit selection masks directly from the engine. However, when building foundational behavioral cloning datasets across millions of matches, running PySC2 across tens of thousands of CPU hours becomes computationally prohibitive. Researchers instead utilize s2protocol to rapidly parse the underlying MPQ containers, bypassing full physics re-simulation to extract tokenized action trajectories for transformer architectures.

### Dataset curation and skill-level filtering

All match replays provided by Blizzard are stripped of personal player identities, eliminating usernames, unique account identifiers, and in-game chat messages. The file headers retain essential statistical and matchmaking metadata, including total game loop duration, player race configurations, game outcome flags, player Actions Per Minute (APM), and player Matchmaking Ratings (MMR).

Because the uncurated 20-million-match API corpus includes games across all competitive divisions, spanning from low-tier amateur games below 1,500 MMR to professional matches exceeding 6,000 MMR, training behavioral cloning models directly on unfiltered archives degrades agent competence. Unfiltered data introduces high noise, blunders, and conflicting tactical responses into policy learning.

To produce competitive artificial intelligence benchmarks, research methodologies such as those developed for AlphaStar implement a two-stage data curation process. First, the raw replay collection is filtered to eliminate games where either participant has an MMR below 3,500, isolating the top 22% of human ladder competitors and yielding approximately 1.4 million games (2.8 million player trajectories). Second, foundational behavioral cloning models trained across this 3,500+ MMR distribution undergo fine-tuning strictly on winning trajectories executed by Grandmaster competitors with MMRs exceeding 6,200. This staged exposure prevents policy collapse during early training while steering the final action distributions toward high-level play.

## Systematic Pipeline Execution and Dataset Construction

Establishing a robust StarCraft II training dataset involves a phased acquisition and engineering pipeline designed to prevent bandwidth and storage bottlenecks.

The initial phase focuses on local pipeline validation using the 64,327 matches in `3.16.1-Pack_1-fix.zip` alongside `Melee.zip` from Blizzard's Akamai CDN. Merging these packages into a local StarCraft II client root and extracting them using `iagreetotheeula` allows researchers to verify feature extraction routines, action masking logic, and tensor formatting on a stable, self-contained baseline without authentication overhead.

Once data loaders and state representation pipelines are verified, scaling to medium-volume training involves ingesting `3.16.1-Pack_2.zip`. Incorporating Pack 2 expands the local corpus to roughly 500,000 games while remaining locked to the 3.16.1 client binary, eliminating the need to manage multi-version execution environments.

For large-scale machine learning initiatives requiring 1 million to 20 million games, the operational pipeline shifts to Blizzard's Game Data API. The engineering team registers an application client on the Blizzard Developer Portal to obtain OAuth credentials, deploys `download_replays.py` targeting stable client versions (such as 4.8.2 through 4.10.0), and streams version-locked batch partitions to distributed storage. The raw replay pool is then parsed using s2protocol or headless Linux client workers (`SC2.4.10.zip`), indexing matches by game duration and player MMR. Enforcing an MMR threshold above 3,500 yields a curated, high-signal corpus of 1.4 million competitive games, establishing the data foundation required for state-of-the-art imitation and offline reinforcement learning.
