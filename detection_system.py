# detection_system.py
import os
import time
import json
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

from config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, COUNT_LINE_Y,
    OVERLAY_X, OVERLAY_Y, OVERLAY_WIDTH, OVERLAY_HEIGHT,
    FONT_SCALE, FONT_THICKNESS, LINE_HEIGHT,
    CLASS_NAMES, OUTPUT_DIR
)
from tracker import CentroidTracker

class VehicleDetectionSystem:
    def __init__(self):
        # Load YOLOv8 model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)

        self.tracker = CentroidTracker()
        self.entry_counts = defaultdict(int)
        self.exit_counts  = defaultdict(int)
        self.counted_ids  = set()
        self.session_start = datetime.now()

    def get_color(self, cid):
        np.random.seed(cid)
        return tuple(np.random.randint(0, 255, 3).tolist())

    def calculate_fps(self, start, frames):
        elapsed = time.time() - start
        return frames / elapsed if elapsed > 0 else 0

    def display_counts(self, frame, fps=0):
        overlay = frame.copy()
        x, y = OVERLAY_X, OVERLAY_Y
        w, h = OVERLAY_WIDTH, OVERLAY_HEIGHT
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        yy = y + LINE_HEIGHT
        # FPS
        cv2.putText(frame, f"FPS:{fps:.1f}", (x + 5, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
        yy += LINE_HEIGHT + 5

        # Entries
        cv2.putText(frame, "IN:", (x + 5, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)
        yy += LINE_HEIGHT
        for cid, cnt in self.entry_counts.items():
            if cnt > 0:
                cv2.putText(frame, f"{CLASS_NAMES[cid]}:{cnt}", (x + 10, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8,
                            self.get_color(cid), 1)
                yy += LINE_HEIGHT - 2

        yy += 5
        # Exits
        cv2.putText(frame, "OUT:", (x + 5, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
        yy += LINE_HEIGHT
        for cid, cnt in self.exit_counts.items():
            if cnt > 0:
                cv2.putText(frame, f"{CLASS_NAMES[cid]}:{cnt}", (x + 10, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8,
                            self.get_color(cid), 1)
                yy += LINE_HEIGHT - 2

        # Net count
        yy += 5
        tot_in  = sum(self.entry_counts.values())
        tot_out = sum(self.exit_counts.values())
        cv2.putText(frame, f"NET:{tot_in - tot_out}", (x + 5, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 0), FONT_THICKNESS)

    def process_frame(self, frame):
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        rects, cids = [], []
        for r in results:
            for b in (r.boxes or []):
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                rects.append((x1, y1, x2, y2))
                cids.append(int(b.cls[0]))

        objects = self.tracker.update(rects, cids)

        for oid, (centroid, cid) in objects.items():
            # update history
            hist = self.tracker.history[oid]
            hist.append(centroid)

            # draw box + label
            for (x1, y1, x2, y2) in rects:
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                if abs(cx - centroid[0]) < 15 and abs(cy - centroid[1]) < 15:
                    color = self.get_color(cid)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    lbl = f"{CLASS_NAMES[cid]} ID:{oid}"
                    (w, h), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                    cv2.putText(frame, lbl, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    break

            # draw short trail
            pts = list(hist)
            for i in range(max(1, len(pts) - 3), len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 1)

            # count crossing
            if len(pts) >= 2:
                py, cy = pts[-2][1], pts[-1][1]
                if py < COUNT_LINE_Y <= cy and (oid, "in") not in self.counted_ids:
                    self.entry_counts[cid] += 1
                    self.counted_ids.add((oid, "in"))
                elif py > COUNT_LINE_Y >= cy and (oid, "out") not in self.counted_ids:
                    self.exit_counts[cid] += 1
                    self.counted_ids.add((oid, "out"))

        # draw counting line
        cv2.line(frame, (0, COUNT_LINE_Y), (frame.shape[1], COUNT_LINE_Y), (0, 0, 255), 2)
        cv2.putText(frame, "COUNT", (frame.shape[1]//2 - 40, COUNT_LINE_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def save_session(self, mode):
        data = {
            "mode": mode,
            "start": self.session_start.isoformat(),
            "end": datetime.now().isoformat(),
            "entries": dict(self.entry_counts),
            "exits": dict(self.exit_counts)
        }
        fn = os.path.join(OUTPUT_DIR, f"{mode}_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(fn, "w") as f:
            json.dump(data, f, indent=2)
        print("Saved session to", fn)

    def detect_video(self, path, window="Video"):
        cap = cv2.VideoCapture(path)
        fc, start, fps = 0, time.time(), 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (960, 540))
            self.process_frame(frame)
            fc += 1
            if fc % 30 == 0:
                fps = self.calculate_fps(start, fc)
            self.display_counts(frame, fps)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self.save_session("video")
        cap.release()
        cv2.destroyAllWindows()

    def detect_webcam(self):
        self.detect_video(0, window="Webcam")

    def detect_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (960, 540))
        self.process_frame(img)
        self.display_counts(img, 0)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.save_session("image")
