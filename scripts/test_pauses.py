import sys
import mpyq
from s2protocol import versions

def main():
    replay_path = sys.argv[1]
    archive = mpyq.MPQArchive(replay_path)
    
    # Read the protocol version
    contents = archive.header['user_data_header']['content']
    header = versions.latest().decode_replay_header(contents)
    protocol = versions.latest()
    
    # Read game events
    # Read game events
    game_events = protocol.decode_replay_game_events(archive.read_file('replay.game.events'))
    
    pause_candidates = {
        'NNet.Game.STriggerSoundLengthSyncEvent',
        'NNet.Game.SSetSyncLoadingTimeEvent',
        'NNet.Game.SSetSyncPlayingTimeEvent',
        'NNet.Game.SDecrementGameTimeRemainingEvent'
    }
    
    for e in game_events:
        if e['_event'] in pause_candidates:
            print("CANDIDATE EVENT:", e)

    
    print("Finished checking events.")

if __name__ == "__main__":
    main()
