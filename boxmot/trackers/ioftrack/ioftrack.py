# Based on ImprAssocTrack and IOF-Tracker paper

import numpy as np
from collections import deque
from pathlib import Path
import cv2 # Added for optical flow

from torch import device # Keep type hint

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc.sof import SOF
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH # Use this KF
from boxmot.trackers.imprassoc.basetrack import BaseTrack, TrackState # Use STrack's base and states
from boxmot.utils.matching import (embedding_distance, iou_distance, 
                                   linear_assignment) # Use boxmot matching utils
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh # Use boxmot coordinate utils
from boxmot.trackers.basetracker import BaseTracker

# Reuse STrack definition from ImprAssocTrack (or assume it's importable)
# Ensure STrack has properties like xyxy, mean, covariance, id, state, etc.
# Minor modification to STrack might be needed if avg_flow needs to be stored,
# but we can compute it on the fly using the predicted bbox.
class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, det, feat=None, feat_history=50): # Increased feat_history default?
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6] # Store original detection index

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.cls_hist = []  # (cls id, freq)
        # self.update_cls(self.cls, self.conf) # Let's call update_cls externally after init if needed

        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history) # Store features
        self.alpha = 0.9 # For feature smoothing

    def update_features(self, feat):
        if feat is None: return # Handle case where feature is None (e.g. low score match)
        feat_norm = np.linalg.norm(feat)
        if feat_norm < 1e-7: return # Avoid division by zero or near-zero
        feat /= feat_norm # Normalize
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            # Ensure smooth_feat is also normalized before smoothing
            smooth_feat_norm = np.linalg.norm(self.smooth_feat)
            if smooth_feat_norm > 1e-7:
                 self.smooth_feat /= smooth_feat_norm
            else: # Handle case where smooth_feat was zero
                 self.smooth_feat = feat # Reset smooth_feat
                 
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            
        self.features.append(feat) # Add current (normalized) feature to history
        # Re-normalize smooth_feat after smoothing
        smooth_feat_norm_after = np.linalg.norm(self.smooth_feat)
        if smooth_feat_norm_after > 1e-7:
             self.smooth_feat /= smooth_feat_norm_after
        else:
             self.smooth_feat = None # Reset if becomes zero

    def update_cls(self, cls, conf):
        # (Simplified - can reuse ImprAssocTrack's version if needed)
        self.cls = cls # Just update to latest class

    def predict(self):
        # Use the shared Kalman filter instance for prediction
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            # Zero out velocity if lost/inactive
            mean_state[4] = 0 
            mean_state[5] = 0
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][4] = 0
                    multi_mean[i][5] = 0
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        # Assuming H is a 2x3 affine matrix
        if len(stracks) > 0 and H is not None:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2] # Rotation/scale part
            # Build 8x8 transformation matrix for the state (xc, yc, w, h, vx, vy, vw, vh)
            # Assuming vx, vy transform like position, and w, h, vw, vh are less affected or handled differently
            # Simplified: Apply R to position (xc, yc) and velocity (vx, vy)
            # More accurate might involve transforming covariance too.
            # Let's use boxmot's approach if available, otherwise a simplified one.
            # This implementation only transforms the position mean based on H.
            # Velocities and covariance are harder to warp correctly without a specific method.
            # BOXMOT's ImprAssocTrack does this warping before prediction. Let's follow that.
            t = H[:2, 2] # Translation part

            for i, mean in enumerate(multi_mean):
                 # Warp center position xc, yc
                 mean[:2] = R @ mean[:2] + t
                 # Optionally warp velocity vx, vy (less common/stable)
                 # mean[4:6] = R @ mean[4:6] 
                 stracks[i].mean = mean
                 # Note: Covariance warping is complex and often omitted or approximated.

    def activate(self, kalman_filter, frame_count):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh) # Use KF's initiate

        self.tracklet_len = 0
        self.state = TrackState.Tracked # Start as tracked (or tentative if needed)
        self.is_activated = True
        self.frame_count = frame_count
        self.start_frame = frame_count
        # Add first feature if available
        if self.curr_feat is not None and len(self.features) == 0:
             self.features.append(self.curr_feat)
             self.smooth_feat = self.curr_feat
        self.update_cls(self.cls, self.conf) # Update class info

    def re_activate(self, new_track, frame_count, new_id=False):
        # Use KF's update method
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh # Update with measurement xywh
        )
        self.update_features(new_track.curr_feat) # Update features
        self.tracklet_len = 0 # Reset tracklet length? Or maybe not? Let's keep it cumulative
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_count = frame_count
        if new_id: # Should not happen with IOF logic usually
            self.id = self.next_id()
        self.conf = new_track.conf
        # self.cls = new_track.cls # Class updated by update_cls
        self.det_ind = new_track.det_ind # Update original det index
        self.update_cls(new_track.cls, new_track.conf) # Update class info

    def update(self, new_track, frame_count):
        """Update a matched track"""
        self.frame_count = frame_count
        self.tracklet_len += 1

        # Update KF state
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh # Update with measurement xywh
        )
        self.update_features(new_track.curr_feat) # Update features

        self.state = TrackState.Tracked
        self.is_activated = True

        self.conf = new_track.conf
        # self.cls = new_track.cls # Class updated by update_cls
        self.det_ind = new_track.det_ind # Update original det index
        self.update_cls(new_track.cls, new_track.conf) # Update class info

    @property
    def pred_xyxy(self):
        """Returns the predicted bounding box `(x1, y1, x2, y2)` from KF state."""
        if self.mean is None:
             # If KF not initialized, return initial bbox? Or raise error?
             # Let's return the initial one for safety, though this shouldn't happen for active tracks
             return xywh2xyxy(self.xywh) 
        ret = self.mean[:4].copy() # KF state is (xc, yc, w, h, ...)
        ret = xywh2xyxy(ret) # Convert to (x1, y1, x2, y2)
        return ret
        
    @property
    def xyxy(self):
        """Returns the *current* bounding box from KF state `(x1, y1, x2, y2)`."""
        # Same as predicted in this setup, as update overwrites mean based on measurement
        return self.pred_xyxy

    @property
    def current_feature(self):
        """Returns the latest smoothed appearance feature."""
        return self.smooth_feat


class IOFTrack(BaseTracker):
    """
    IOF-Tracker: Implements the two-stage spatial-temporal fusion tracking algorithm.

    Args:
        reid_weights (Path): Path to the model weights for ReID.
        device (device): Torch device.
        half (bool): Use half-precision for ReID.
        per_class (bool): Perform tracking per class.
        track_high_thresh (float): Confidence threshold for high-score detections (Stage 1).
        track_low_thresh (float): Confidence threshold for low-score detections (Stage 2).
        new_track_thresh (float): Confidence threshold to initialize a new track from high-score unmatched detections.
        iou_dist_thresh (float): R-IoU distance threshold (1 - IoU threshold). Matches with R-IoU > thresh are penalized/gated. Example: 0.7 means IoU < 0.3 is penalized.
        of_dist_thresh (float): Euclidean distance threshold for average optical flow vectors.
        reid_dist_thresh (float): Appearance (cosine) distance threshold (theta_emb).
        fusion_alpha (float): Weight for R-IoU cost in fusion (Eq 9). (1 - alpha) is weight for OF cost.
        theta_fusion (float): Fused spatial-temporal cost threshold used in Eq 10 gating.
        final_match_thresh (float): Final cost threshold applied after Hungarian assignment.
        track_buffer (int): Number of frames to keep a track alive (lost).
        cmc_method (str): Camera motion compensation method ("sparseOptFlow", "orb", "ecc", None).
        frame_rate (int): Frame rate for track buffer scaling.
        with_reid (bool): Enable ReID feature usage.
    """
    def __init__(
        self,
        reid_weights: Path = Path("osnet_x0_25_msmt17.pt"), # Default weights
        device: device = device("cuda:0"),
        half: bool = True,
        per_class: bool = False,
        # IOF specific thresholds and params
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7, # Threshold to init new tracks from high score dets
        iou_dist_thresh: float = 0.7,  # R-IoU threshold (1 - IoU_thresh)
        of_dist_thresh: float = 10.0,  # Optical flow distance threshold (needs tuning)
        reid_dist_thresh: float = 0.5, # Appearance cost threshold (theta_emb)
        fusion_alpha: float = 0.8,     # Weight for R-IoU in fusion
        theta_fusion: float = 0.5,     # Fused ST cost threshold for gating (Eq 10)
        final_match_thresh: float = 0.7, # Final cost threshold after Hungarian
        # General params from BaseTracker/ImprAssocTrack
        track_buffer: int = 30,
        cmc_method: str = "sparseOptFlow",
        frame_rate=30,
        with_reid: bool = True
    ):
        super().__init__(per_class=per_class)
        self.active_tracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh # Ensure this is set

        # IOF specific thresholds
        self.iou_dist_thresh = iou_dist_thresh
        self.of_dist_thresh = of_dist_thresh
        self.reid_dist_thresh = reid_dist_thresh # theta_emb
        self.fusion_alpha = fusion_alpha
        self.theta_fusion = theta_fusion
        self.final_match_thresh = final_match_thresh # Renamed from match_thresh for clarity

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH() # Use boxmot's KF

        # ReID module
        self.with_reid = with_reid
        if self.with_reid:
            rab = ReidAutoBackend(
                weights=reid_weights, device=device, half=half
            )
            self.model = rab.get_backend()
        
        # CMC module
        self.cmc = SOF()

        # Optical Flow related state
        self.prev_gray = None
        self.of_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
                              poly_n=5, poly_sigma=1.2, flags=0) # Farneback params
        
    def _calculate_image_optical_flow(self, gray):
        """Calculates optical flow between the previous grayscale images and the current one."""
        # First lets get a local copy of the previous frame
        prev_gray = self.prev_gray
        # Update the previous frame with the current one
        self.prev_gray = gray.copy() # Store current frame for next call

        if prev_gray is None:

            return np.zeros_like(gray, dtype=np.float32)
        try:
            # Ensure gray and prev_gray have the same dimensions
            if gray.shape != prev_gray.shape:
                # Handle potential resize issues, e.g., log a warning or resize prev_gray
                # For now, return empty flow if shapes mismatch significantly
                print(f"Warning: Frame shape mismatch in optical flow. Prev: {self.prev_gray.shape}, Curr: {gray.shape}")
                return np.zeros_like(gray, dtype=np.float32)
            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, **self.of_params)
        except cv2.error as e:
            print(f"OpenCV error calculating Farneback flow: {e}")
            # Return zeros if flow calculation fails
            return np.zeros_like(gray, dtype=np.float32)
        
        # Return the flow vector
        return flow

    def _calculate_avg_optical_flow_boxes(self, flow, bboxes):
        """Calculates average optical flow vector within bounding boxes."""
        # Expects bboxes in [x1, y1, x2, y2] format
        if flow is None or not isinstance(bboxes, np.ndarray) or bboxes.shape[0] == 0:
             # Return empty array with correct shape (N, 2)
             return np.empty((0, 2), dtype=np.float32) 

        avg_flows = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            # Clamp coordinates to be within frame dimensions
            h, w = flow.shape[:2] # Get height and width of the flow field
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w)) # Use w, h for slicing upper bound
            y2 = max(0, min(y2, h))

            if x1 >= x2 or y1 >= y2:
                 avg_flows.append(np.array([0.0, 0.0], dtype=np.float32)) # Handle zero-area boxes
                 continue

            box_flow = flow[y1:y2, x1:x2]
            if box_flow.size == 0:
                 avg_flows.append(np.array([0.0, 0.0], dtype=np.float32)) # Handle empty flow region
                 continue
                 
            # Calculate mean flow vector (dx, dy) for the box
            avg_flow_vec = np.mean(box_flow, axis=(0, 1), dtype=np.float32)
            avg_flows.append(avg_flow_vec)
            
        return np.array(avg_flows, dtype=np.float32) if avg_flows else np.empty((0, 2), dtype=np.float32)

    def _calculate_of_cost(self, track_bboxes_pred, det_bboxes, current_gray):
        """ Calculates the pairwise Euclidean distance cost based on average optical flow."""
        n_trk = len(track_bboxes_pred)
        n_det = len(det_bboxes)
        cost_of = np.full((n_trk, n_det), self.of_dist_thresh * 2.0) # Default high cost

        if n_trk == 0 or n_det == 0 or self.prev_gray is None:
             return cost_of # Return default if no tracks, dets, or previous frame
        
        # Calculate optical flow for the current frame
        current_flow = self._calculate_image_optical_flow(current_gray)

        # Calculate average flow for predicted track locations and detection locations
        trk_avg_flows = self._calculate_avg_optical_flow_boxes(current_flow, track_bboxes_pred)
        det_avg_flows = self._calculate_avg_optical_flow_boxes(current_flow, det_bboxes)

        # Ensure flows were calculated and shapes match number of boxes
        if trk_avg_flows.shape[0] != n_trk or det_avg_flows.shape[0] != n_det:
             print(f"Warning: Optical flow calculation mismatch. Tracks: {n_trk} vs {trk_avg_flows.shape[0]}, Dets: {n_det} vs {det_avg_flows.shape[0]}")
             return cost_of # Return default if calculation failed

        # Calculate pairwise Euclidean distance
        for i in range(n_trk):
             for j in range(n_det):
                 # Euclidean distance between the two average flow vectors
                 flow_dist = np.linalg.norm(trk_avg_flows[i] - det_avg_flows[j])
                 cost_of[i, j] = flow_dist
                 
        return cost_of

    def _fuse_costs_stage1(self, cost_iou, cost_of, cost_reid):
        """ Fuse costs for Stage 1 (R-IoU + OF + ReID) using Eq 10 logic. """
        
        # Calculate d_fusion (fused spatial-temporal cost)
        # Note: cost_iou from boxmot's iou_distance is R-IoU cost (1 - IoU)
        d_fusion = (self.fusion_alpha * cost_iou + 
                    (1 - self.fusion_alpha) * cost_of)

        # Apply Eq 10 gating logic
        theta_emb = self.reid_dist_thresh 
        theta_fusion = self.theta_fusion 

        condition = (cost_reid < theta_emb) & (d_fusion < theta_fusion)
        d_cos_hat = np.full_like(cost_reid, 1.0 + 1e-6) # Initialize with high cost (slightly > 1)
        d_cos_hat[condition] = cost_reid[condition]

        # Final cost is element-wise minimum
        cost_final = np.minimum(d_fusion, d_cos_hat)
        
        return cost_final
        
    def _fuse_costs_stage2(self, cost_iou, cost_of):
        """ Fuse costs for Stage 2 (R-IoU + OF only) using Eq 9 logic. """
        # Note: cost_iou from boxmot's iou_distance is R-IoU cost (1 - IoU)
        d_fusion = (self.fusion_alpha * cost_iou + 
                    (1 - self.fusion_alpha) * cost_of)
        return d_fusion

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Processes a new frame using the IOF-Tracker two-stage logic.
        Args:
            dets (np.ndarray): Detections [N, 6] (x1, y1, x2, y2, conf, cls).
            img (np.ndarray): Current frame (BGR format).
            embs (np.ndarray, optional): Precomputed embeddings for detections.
        Returns:
            np.ndarray: Tracked objects [M, 7] (x1, y1, x2, y2, track_id, conf, cls, det_idx).
        """
        self.check_inputs(dets, img) # Basic checks
        self.frame_count += 1
        
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # --- 0. Preprocessing ---
        # Add detection index to detections array
        if dets.shape[1] == 6:
            dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)]) # Now [N, 7] (..., det_idx)

        # Get current frame in grayscale for optical flow
        current_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Separate detections by score
        confs = dets[:, 4]
        high_score_mask = confs >= self.track_high_thresh
        low_score_mask = (~high_score_mask) & (confs >= self.track_low_thresh)
        
        dets_high = dets[high_score_mask]
        dets_low = dets[low_score_mask]

        # Extract ReID features for high-score detections
        features_high = None
        if self.with_reid and len(dets_high) > 0:
            if embs is not None:
                # Assume embs correspond to the *original* dets array order
                original_indices_high = dets_high[:, 6].astype(int)
                features_high = embs[original_indices_high]
            else:
                features_high = self.model.get_features(dets_high[:, :4], img) # BGR image

        # Create STrack objects for high-score detections
        detections_high = []
        if len(dets_high) > 0:
            for i, det in enumerate(dets_high):
                feat = features_high[i] if features_high is not None else None
                detections_high.append(STrack(det, feat)) # Pass feature here

        # Create STrack objects for low-score detections (no features needed)
        detections_low = [STrack(det) for det in dets_low] if len(dets_low) > 0 else []

        # Separate active tracks from tentative/unconfirmed ones (though IOF doesn't really have tentative)
        active_tracks = []
        unconfirmed_tracks = [] # Keep this concept? IOF paper doesn't mention. Assume all non-lost are active.
        for track in self.active_tracks:
             if not track.is_activated: # Should generally be true after first frame
                 unconfirmed_tracks.append(track) # Should be empty usually
             else:
                 active_tracks.append(track)
                 
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # --- 1. Motion Prediction & Compensation ---
        STrack.multi_predict(strack_pool)
        # STrack.multi_predict(unconfirmed_tracks) # Predict unconfirmed if any

        # Apply CMC (Camera Motion Compensation)
        warp = None
        if self.cmc is not None and len(dets) > 0:
             # Use high confidence dets for CMC usually
             warp = self.cmc.apply(img, dets_high if len(dets_high)>0 else dets) 
        if warp is not None:
             STrack.multi_gmc(strack_pool, warp)
             # STrack.multi_gmc(unconfirmed_tracks, warp)

        # Get predicted bboxes for cost calculation [N, 4] (x1,y1,x2,y2)
        trk_pred_bboxes = np.array([t.pred_xyxy for t in strack_pool])
        det_high_bboxes = np.array([d.xyxy for d in detections_high])
        det_low_bboxes = np.array([d.xyxy for d in detections_low])

        # --- 2. Association Stage 1 (High Score Detections) ---
        if len(strack_pool) > 0 and len(detections_high) > 0:
             # Calculate costs: R-IoU, Optical Flow, ReID
             cost_iou_s1 = iou_distance(strack_pool, detections_high) # R-IoU cost
             cost_of_s1 = self._calculate_of_cost(trk_pred_bboxes, det_high_bboxes, current_gray)
             cost_reid_s1 = embedding_distance(strack_pool, detections_high) if self.with_reid else np.full_like(cost_iou_s1, 1.0)
             
             # Fuse costs for stage 1
             cost_fused_s1 = self._fuse_costs_stage1(cost_iou_s1, cost_of_s1, cost_reid_s1)

             # Perform linear assignment
             matches_s1, um_trk_s1_indices, um_det_s1_indices = linear_assignment(
                 cost_fused_s1, thresh=self.final_match_thresh # Use the final threshold
             )
             
             # Update matched tracks
             for trk_idx, det_idx in matches_s1:
                 track = strack_pool[trk_idx]
                 det = detections_high[det_idx]
                 if track.state == TrackState.Tracked:
                     track.update(det, self.frame_count)
                     activated_stracks.append(track) # Keep track of updated active tracks
                 else: # Reactivate lost track
                     track.re_activate(det, self.frame_count)
                     refind_stracks.append(track) # Keep track of reactivated tracks
                 # Remove updated track/det from unmatched lists
                 # Note: um_trk_s1_indices and um_det_s1_indices already contain only the unmatched
        else: # No tracks or no high score detections
             matches_s1 = []
             um_trk_s1_indices = list(range(len(strack_pool)))
             um_det_s1_indices = list(range(len(detections_high)))

        # --- 3. Association Stage 2 (Low Score Detections with Remaining Tracks) ---
        # Get tracks that were not matched in stage 1
        remaining_tracklets_s1 = [strack_pool[i] for i in um_trk_s1_indices]
        # Get predicted bboxes for remaining tracks
        remaining_trk_pred_bboxes = np.array([t.pred_xyxy for t in remaining_tracklets_s1])

        if len(remaining_tracklets_s1) > 0 and len(detections_low) > 0:
             # Calculate costs: R-IoU, Optical Flow (No ReID for stage 2)
             cost_iou_s2 = iou_distance(remaining_tracklets_s1, detections_low) # R-IoU cost
             cost_of_s2 = self._calculate_of_cost(remaining_trk_pred_bboxes, det_low_bboxes, current_gray)
             
             # Fuse costs for stage 2
             cost_fused_s2 = self._fuse_costs_stage2(cost_iou_s2, cost_of_s2)
             
             # Perform linear assignment
             # Use a potentially different threshold for stage 2? Paper doesn't specify. Use same for now.
             matches_s2, um_trk_s2_indices, um_det_s2_indices = linear_assignment(
                 cost_fused_s2, thresh=self.final_match_thresh 
             )
             
             # Update matched tracks (pass None for feature)
             for trk_idx, det_idx in matches_s2:
                 track = remaining_tracklets_s1[trk_idx]
                 det = detections_low[det_idx]
                 if track.state == TrackState.Tracked:
                     track.update(det, self.frame_count) # Update state, KF, but maybe not feature
                     # track.update_features(None) # Explicitly avoid feature update
                     activated_stracks.append(track)
                 else: # Reactivate lost track with low score detection
                     track.re_activate(det, self.frame_count)
                     # track.update_features(None)
                     refind_stracks.append(track)
                 # Remove updated track/det from unmatched lists
        else: # No remaining tracks or no low score detections
            matches_s2 = []
            um_trk_s2_indices = list(range(len(remaining_tracklets_s1)))
            um_det_s2_indices = list(range(len(detections_low)))

        # --- 4. Handle Remaining Tracks (Mark as Lost) ---
        # Tracks unmatched in Stage 1 and also unmatched in Stage 2
        final_unmatched_trk_pool_indices = [um_trk_s1_indices[i] for i in um_trk_s2_indices]
        for trk_pool_idx in final_unmatched_trk_pool_indices:
            track = strack_pool[trk_pool_idx]
            # Only mark as lost if it wasn't already lost and didn't get reactivated
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # --- 5. Initialize New Tracks (From Unmatched HIGH Score Detections) ---
        for det_idx in um_det_s1_indices:
            track = detections_high[det_idx]
            # Check confidence against the new track threshold
            if track.conf >= self.new_track_thresh:
                track.activate(self.kalman_filter, self.frame_count)
                activated_stracks.append(track)
                # Feature was already added during STrack creation for high dets

        # --- 6. Update State & Cleanup ---
        # Remove old lost tracks
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update track lists
        # Consolidate active tracks: original active + newly activated + refound
        # Filter out tracks that were matched but maybe ended up in lost_stracks incorrectly?
        current_active_ids = {t.id for t in activated_stracks + refind_stracks}
        self.active_tracks = [t for t in self.active_tracks if t.id in current_active_ids]
        # Add newly activated and refound tracks
        self.active_tracks = joint_stracks(self.active_tracks, [t for t in activated_stracks if t.start_frame == self.frame_count]) # Add brand new tracks
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks) # Add reactivated tracks

        # Update lost tracks list
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks) # Remove reactivated
        self.lost_stracks.extend(lost_stracks) # Add newly lost
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks) # Remove expired

        # Remove duplicates (e.g., if a track was in lost and got reactivated)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )
        
        self.removed_stracks.extend(removed_stracks) # Keep removed tracks history

        # --- 7. Prepare Output ---
        # Output only currently 'Tracked' state tracks
        output_tracks = [track for track in self.active_tracks if track.state == TrackState.Tracked]
        outputs = []
        for t in output_tracks:
             output = []
             output.extend(t.xyxy)      # x1, y1, x2, y2
             output.append(t.id)        # track_id
             output.append(t.conf)      # confidence score
             output.append(t.cls)       # class id
             output.append(t.det_ind)   # original detection index
             outputs.append(output)

        if len(outputs) > 0:
             outputs = np.asarray(outputs)
        else:
             outputs = np.empty((0, 8)) # Match output shape [x1,y1,x2,y2,id,conf,cls,det_idx]
             
        # --- 8. Update Previous Frame for Optical Flow ---
        self.prev_gray = current_gray

        # print(f"Frame {self.frame_count}: Active={len(self.active_tracks)}, Lost={len(self.lost_stracks)}, Output={len(outputs)}")
        return outputs

# Helper functions (reuse from ImprAssocTrack or define here if needed)
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
    # Simple check based on ID should be sufficient if state management is correct
    idsa = {t.id for t in stracksa}
    idsb = {t.id for t in stracksb}
    common_ids = idsa.intersection(idsb)
    
    # Keep the version from stracksa (active) if ID is duplicated
    resb = [t for t in stracksb if t.id not in common_ids]
    
    # Check IoU based overlap for non-ID duplicates (less likely but possible)
    # Using ImprAssoc's version for robustness
    pdist = iou_distance(stracksa, resb) # Compare active with filtered lost
    pairs = np.where(pdist < 0.15) # Find overlaps
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        # Keep the track that has been active longer or is currently active
        # Assuming stracksa are generally more up-to-date if matched
        dupb.append(q) # Mark the one in resb (lost) for removal
        
    # Final lists
    resa = stracksa # Keep all active tracks
    resb = [t for i, t in enumerate(resb) if i not in dupb] # Keep lost tracks not marked duplicate
    return resa, resb