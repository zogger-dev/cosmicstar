#!/usr/bin/env python
import sys
import os
import mpyq
from s2protocol import versions
import pprint


def dump_events(replay_path, output_dir):
    archive = mpyq.MPQArchive(replay_path)
    protocol = versions.latest()

    os.makedirs(output_dir, exist_ok=True)

    # Dump header
    contents = archive.header['user_data_header']['content']
    header = versions.latest().decode_replay_header(contents)
    with open(os.path.join(output_dir, "header_events.txt"), "w", encoding="utf-8") as f:
        f.write(pprint.pformat(header) + "\n")

    # Dump game events
    game_events = protocol.decode_replay_game_events(
        archive.read_file("replay.game.events")
    )
    with open(os.path.join(output_dir, "game_events.txt"), "w", encoding="utf-8") as f:
        for event in game_events:
            f.write(pprint.pformat(event) + "\n")

    # Dump message events
    message_events = protocol.decode_replay_message_events(
        archive.read_file("replay.message.events")
    )
    with open(
        os.path.join(output_dir, "message_events.txt"), "w", encoding="utf-8"
    ) as f:
        for event in message_events:
            f.write(pprint.pformat(event) + "\n")

    # Dump tracker events
    tracker_events = protocol.decode_replay_tracker_events(
        archive.read_file("replay.tracker.events")
    )
    with open(
        os.path.join(output_dir, "tracker_events.txt"), "w", encoding="utf-8"
    ) as f:
        for event in tracker_events:
            f.write(pprint.pformat(event) + "\n")

    # Dump init data
    init_data = protocol.decode_replay_initdata(archive.read_file("replay.initData"))
    with open(os.path.join(output_dir, "init_data.txt"), "w", encoding="utf-8") as f:
        f.write(pprint.pformat(init_data) + "\n")

    # Dump details
    details = protocol.decode_replay_details(archive.read_file("replay.details"))
    with open(os.path.join(output_dir, "details.txt"), "w", encoding="utf-8") as f:
        f.write(pprint.pformat(details) + "\n")

    # Dump attributes
    attributes = protocol.decode_replay_attributes_events(
        archive.read_file("replay.attributes.events")
    )
    with open(
        os.path.join(output_dir, "attributes_events.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(pprint.pformat(attributes) + "\n")

    # Dump sync events
    try:
        sync_events = protocol.decode_replay_sync_events(
            archive.read_file("replay.sync.events")
        )
        with open(
            os.path.join(output_dir, "sync_events.txt"), "w", encoding="utf-8"
        ) as f:
            for event in sync_events:
                f.write(pprint.pformat(event) + "\n")
    except Exception as e:
        print(f"Could not dump sync events: {e}")

    # Dump all raw files
    raw_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    for filename in archive.files:
        try:
            fname_str = filename.decode('utf-8') if isinstance(filename, bytes) else filename
            with open(os.path.join(raw_dir, fname_str), 'wb') as f:
                f.write(archive.read_file(filename))
        except Exception as e:
            print(f"Could not dump raw file {filename}: {e}")

    print(f"Dumped events to {output_dir}")


if __name__ == "__main__":
    replay_path = sys.argv[1]
    output_dir = sys.argv[2]
    dump_events(replay_path, output_dir)
