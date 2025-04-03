# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '12.0.4'

from boxmot.postprocessing.gsi import gsi
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.trackers.hybridsort.hybridsort import HybridSort
from boxmot.trackers.ocsort.ocsort import OcSort
from boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.trackers.imprassoc.imprassoctrack import ImprAssocTrack
from boxmot.trackers.boosttrack.boosttrack import BoostTrack
from boxmot.trackers.ioftrack.ioftrack import IOFTrack


TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort', 'imprassoc', 'boosttrack', 'ioftrack']

__all__ = ("__version__",
           "StrongSort", "OcSort", "ByteTrack", "BotSort", "DeepOcSort", "HybridSort", "ImprAssocTrack", "BoostTrack", "IOFTrack",
           "create_tracker", "get_tracker_config", "gsi")
