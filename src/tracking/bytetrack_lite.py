from dataclasses import dataclass, field

import numpy as np

from src.tracking.iou_tracker import iou_one


@dataclass
class ByteTrackLiteTrack:
    track_id: int
    bbox: np.ndarray
    class_id: int
    score: float
    age: int = 0
    hits: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))


class ByteTrackLite:
    """ByteTrack-style two-stage IoU association for educational demos.

    It first matches active tracks to high-confidence detections, then uses
    low-confidence detections to keep unmatched tracks alive. This is not the
    full ByteTrack implementation because it does not include a Kalman filter.
    """

    def __init__(
        self,
        high_thresh=0.35,
        low_thresh=0.08,
        new_track_thresh=0.45,
        iou_thresh=0.3,
        max_age=30,
        min_hits=1,
        smooth=0.0,
        use_motion=False,
        display_max_age=0,
    ):
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.smooth = float(smooth)
        self.use_motion = bool(use_motion)
        self.display_max_age = int(display_max_age)
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        boxes = np.asarray(detections.get("boxes", []), dtype=np.float32)
        scores = np.asarray(detections.get("scores", []), dtype=np.float32)
        labels = np.asarray(detections.get("labels", []), dtype=np.int64)

        for track in self.tracks:
            track.age += 1

        high = np.where(scores >= self.high_thresh)[0]
        low = np.where((scores >= self.low_thresh) & (scores < self.high_thresh))[0]
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_high = self._match(unmatched_tracks, high, boxes, scores, labels)
        self._match(unmatched_tracks, low, boxes, scores, labels)

        for det_idx in unmatched_high:
            if scores[det_idx] >= self.new_track_thresh:
                self.tracks.append(ByteTrackLiteTrack(self.next_id, boxes[det_idx], int(labels[det_idx]), float(scores[det_idx])))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [
            {"track_id": t.track_id, "bbox": t.bbox.copy(), "class_id": t.class_id, "score": t.score}
            for t in self.tracks
            if t.age <= self.display_max_age and t.hits >= self.min_hits
        ]

    def _match(self, unmatched_tracks, det_indices, boxes, scores, labels):
        unmatched_dets = set(int(i) for i in det_indices)
        if not unmatched_tracks or not unmatched_dets:
            return unmatched_dets

        candidates = []
        for track_idx in sorted(unmatched_tracks):
            track = self.tracks[track_idx]
            dets = np.asarray(sorted(i for i in unmatched_dets if labels[i] == track.class_id), dtype=np.int64)
            if len(dets) == 0:
                continue
            ious = iou_one(self._matching_box(track), boxes[dets])
            for det_idx, iou in zip(dets, ious):
                if iou >= self.iou_thresh:
                    candidates.append((float(iou), track_idx, int(det_idx)))

        candidates.sort(reverse=True)
        used_tracks = set()
        used_dets = set()
        for _, track_idx, det_idx in candidates:
            if track_idx in used_tracks or det_idx in used_dets:
                continue
            track = self.tracks[track_idx]
            previous = track.bbox.copy()
            detected = boxes[det_idx].astype(np.float32)
            if self.smooth > 0.0:
                reference = self._matching_box(track)
                track.bbox = (self.smooth * reference + (1.0 - self.smooth) * detected).astype(np.float32)
            else:
                track.bbox = detected
            track.velocity = (track.bbox - previous).astype(np.float32)
            track.score = float(scores[det_idx])
            track.class_id = int(labels[det_idx])
            track.age = 0
            track.hits += 1
            used_tracks.add(track_idx)
            used_dets.add(det_idx)

        unmatched_tracks.difference_update(used_tracks)
        unmatched_dets.difference_update(used_dets)
        return unmatched_dets

    def _matching_box(self, track):
        if not self.use_motion:
            return track.bbox
        return (track.bbox + track.velocity).astype(np.float32)
