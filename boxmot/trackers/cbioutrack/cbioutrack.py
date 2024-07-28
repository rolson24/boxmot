# Raif Olson 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from collections import deque

from boxmot.trackers.cbioutrack.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils import PerClassDecorator


class STrack(BaseTrack):

    def __init__(self, det, feat=None, feat_history=50, max_obs=50, motion_history=5):
        # wait activate
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.max_obs=max_obs
        # self.kalman_filter = None
        self.is_activated = False
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(self.cls, self.conf)
        # self.history_observations = deque([], maxlen=self.max_obs)

        self.tracklet_len = 0

        # self.smooth_feat = None
        # self.curr_feat = None
        # if feat is not None:
        #     self.update_features(feat)
        # self.features = deque([], maxlen=feat_history)
        # self.alpha = 0.9

        self.vx_buffer = deque([], maxlen=motion_history)
        self.vy_buffer = deque([], maxlen=motion_history)
        self.vw_buffer = deque([], maxlen=motion_history)
        self.vh_buffer = deque([], maxlen=motion_history)

        self.motion_mean = np.zeros(4)

    # def update_features(self, feat):
    #     feat /= np.linalg.norm(feat)
    #     self.curr_feat = feat
    #     if self.smooth_feat is None:
    #         self.smooth_feat = feat
    #     else:
    #         self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
    #     self.features.append(feat)
    #     self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, conf):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += conf
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, conf])
                self.cls = cls
        else:
            self.cls_hist.append([cls, conf])
            self.cls = cls

    def predict(self):
        self.xywh += self.motion_mean

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            for st in stracks:
                st.predict()

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.id = self.next_id()

        self.mean[0] = sum(self.vx_buffer) / self.vx_buffer.qsize()
        self.mean[1] = sum(self.vy_buffer) / self.vy_buffer.qsize()
        self.mean[2] = sum(self.vw_buffer) / self.vw_buffer.qsize()
        self.mean[3] = sum(self.vh_buffer) / self.vh_buffer.qsize()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        xywh = new_track.xywh
        self.vx_buffer.append(xywh[0])
        self.vy_buffer.append(xywh[1])
        self.vw_buffer.append(xywh[2])
        self.vh_buffer.append(xywh[3])

        self.mean[0] = sum(self.vx_buffer) / self.vx_buffer.qsize()
        self.mean[1] = sum(self.vy_buffer) / self.vy_buffer.qsize()
        self.mean[2] = sum(self.vw_buffer) / self.vw_buffer.qsize()
        self.mean[3] = sum(self.vh_buffer) / self.vh_buffer.qsize()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        xywh = new_track.xywh
        self.vx_buffer.append(xywh[0])
        self.vy_buffer.append(xywh[1])
        self.vw_buffer.append(xywh[2])
        self.vh_buffer.append(xywh[3])

        self.mean[0] = sum(self.vx_buffer) / self.vx_buffer.qsize()
        self.mean[1] = sum(self.vy_buffer) / self.vy_buffer.qsize()
        self.mean[2] = sum(self.vw_buffer) / self.vw_buffer.qsize()
        self.mean[3] = sum(self.vh_buffer) / self.vh_buffer.qsize()

        self.state = TrackState.Tracked
        self.is_activated = True

        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, a, h)
            ret[2] *= ret[3]  # (xc, yc, a, h)  -->  (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret


class CBIoUTrack(BaseTracker):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        per_class=False,
        # track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        b1,
        b2,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        second_match_thresh: float = 0.6,
        # proximity_thresh: float = 0.5,
        # appearance_thresh: float = 0.25,
        # cmc_method: str = "sof",
        frame_rate=30,
        fuse_first_associate: bool = False,
        # with_reid: bool = True,
    ):
        super().__init__()
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class = per_class
        # self.track_high_thresh = track_high_thresh
        # self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)

        self.fuse_first_associate = fuse_first_associate

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])

        # Remove bad detections
        confs = dets[:, 4]

        # # find second round association detections
        # second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        # dets_second = dets[second_mask]

        # find first round association detections
        conf_mask = confs > self.track_low_thresh
        dets_first = dets[conf_mask]

        # """Extract embeddings """
        # # appearance descriptor extraction
        # if self.with_reid:
        #     if embs is not None:
        #         features_high = embs
        #     else:
        #         # (Ndets x X) [512, 1024, 2048]
        #         features_high = self.model.get_features(dets_first[:, 0:4], img)

        if len(dets) > 0:
                detections = [STrack(det, max_obs=self.max_obs) for (det) in np.array(dets_first)]
        else:
            detections = []

        """ Add newly detected tracklets to active_tracks"""
        unconfirmed = []
        active_tracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.cmc.apply(img, dets_first)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high conf detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
          ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [STrack(dets_second, max_obs=self.max_obs) for dets_second in dets_second]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh

        ious_dists = fuse_score(ious_dists, detections)
        
        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_count)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_age:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        output_stracks = [track for track in self.active_tracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.conf)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)

        outputs = np.asarray(outputs)
        return outputs


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
