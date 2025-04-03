from boxmot import (
    StrongSort, BotSort, DeepOcSort, OcSort, ByteTrack, ImprAssocTrack, BoostTrack, IOFTrack, get_tracker_config, create_tracker,
)

MOTION_N_APPEARANCE_TRACKING_NAMES = ['botsort', 'deepocsort', 'strongsort', 'imprassoc', 'boosttrack', 'ioftrack']
MOTION_ONLY_TRACKING_NAMES = ['ocsort', 'bytetrack']

MOTION_N_APPEARANCE_TRACKING_METHODS=[StrongSort, BotSort, DeepOcSort, ImprAssocTrack, BoostTrack, IOFTrack]
MOTION_ONLY_TRACKING_METHODS=[OcSort, ByteTrack]

ALL_TRACKERS = ['botsort', 'deepocsort', 'ocsort', 'bytetrack', 'strongsort', 'imprassoc', 'boosttrack', 'ioftrack']
PER_CLASS_TRACKERS = ['botsort', 'deepocsort', 'ocsort', 'bytetrack', 'imprassoc', 'ioftrack']