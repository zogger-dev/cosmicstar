import sys
import mpyq
from s2protocol import versions

def examine(replay_path, target_loop):
    archive = mpyq.MPQArchive(replay_path)
    protocol = versions.latest()
    
    events = protocol.decode_replay_game_events(archive.read_file('replay.game.events'))
    
    print(f"Events around loop {target_loop}:")
    for event in events:
        loop = event.get('_gameloop', 0)
        if target_loop - 20 <= loop <= target_loop + 20:
            print(f"Loop {loop}: {event['_event']} (User {event.get('_userid')})")
            if 'm_speed' in str(event):
                print(event)

if __name__ == "__main__":
    examine(sys.argv[1], int(sys.argv[2]))
