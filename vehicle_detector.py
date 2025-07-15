import cv2, os, json, time
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
from centroid_tracker import CentroidTracker
import numpy as np

class VehicleDetectionSystem:
    def __init__(self):
        self.model = YOLO("runs2/detect/traffic_detector/weights/best_2.pt")
        self.class_names = {
            0: 'ambulance', 1: 'auto', 2: 'bicycle',
            3: 'bus', 4: 'car', 5: 'motorbike', 6: 'truck'
        }
        self.conf_threshold = 0.5
        self.session_start = datetime.now()
        self.ct = CentroidTracker()
        self.entry_counts = defaultdict(int)
        self.exit_counts = defaultdict(int)
        self.counted_ids = set()
        self.track_history = {}
        self.line_y = 300

    def get_color(self, cid):
        np.random.seed(cid)
        return tuple(np.random.randint(0,255,3).tolist())

    def calculate_fps(self, start, frames):
        elapsed = time.time() - start
        return frames / elapsed if elapsed > 0 else 0

    def display_counts_on_frame(self, frame, fps):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10,10), (150,250), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        y = 25
        fs, th, lh = 0.5, 1, 18
        cv2.putText(frame, f"FPS:{fps:.1f}", (15,y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,255), th)
        y += lh + 5
        cv2.putText(frame, "ENTRIES:", (15,y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th)
        y += lh
        for cid, cnt in self.entry_counts.items():
            if cnt > 0:
                cv2.putText(frame, f"{self.class_names[cid]}:{cnt}", (20,y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.get_color(cid), 1)
                y += 15
        y += 5
        cv2.putText(frame, "EXITS:", (15,y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,255), th)
        y += lh
        for cid, cnt in self.exit_counts.items():
            if cnt > 0:
                cv2.putText(frame, f"{self.class_names[cid]}:{cnt}", (20,y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.get_color(cid), 1)
                y += 15
        y += 5
        total_in = sum(self.entry_counts.values())
        total_out = sum(self.exit_counts.values())
        cv2.putText(frame, f"IN:{total_in}", (15,y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,255), th)
        y += lh
        cv2.putText(frame, f"OUT:{total_out}", (15,y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,0,255), th)
        y += lh
        cv2.putText(frame, f"NET:{total_in-total_out}", (15,y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,0), th)

    def process_detections_and_track(self, frame):
        res = self.model(frame, conf=self.conf_threshold, verbose=False)
        rects, cids = [], []
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cid = int(b.cls[0])
                rects.append((x1, y1, x2, y2))
                cids.append(cid)
        objs = self.ct.update(rects, cids)
        for oid, (centroid, cid) in objs.items():
            if oid not in self.track_history:
                self.track_history[oid] = deque(maxlen=8)
            self.track_history[oid].append(centroid)
            for (x1,y1,x2,y2) in rects:
                rc = ((x1+x2)//2, (y1+y2)//2)
                if abs(rc[0]-centroid[0])<15 and abs(rc[1]-centroid[1])<15:
                    col = self.get_color(cid)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                    lbl = f"{self.class_names[cid]} ID:{oid}"
                    (w,h), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(frame, (x1,y1-h-5), (x1+w,y1), col, -1)
                    cv2.putText(frame, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                    break
            pts = list(self.track_history[oid])
            for i in range(max(1, len(pts)-3), len(pts)):
                cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), (0,255,255), 1)
            if len(pts) >= 2:
                py, cy = pts[-2][1], pts[-1][1]
                if py < self.line_y <= cy and (oid, 'in') not in self.counted_ids:
                    self.entry_counts[cid] += 1
                    self.counted_ids.add((oid, 'in'))
                elif py > self.line_y >= cy and (oid, 'out') not in self.counted_ids:
                    self.exit_counts[cid] += 1
                    self.counted_ids.add((oid, 'out'))
        cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (0,0,255), 2)
        cv2.putText(frame, "COUNTING LINE", (frame.shape[1]//2 - 80, self.line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    def video_detection(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Cannot open video"); return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", 960, 540)
        fc, start, fps = 0, time.time(), 0
        while True:
            ret, frm = cap.read()
            if not ret: break
            frm = cv2.resize(frm, (960,540))
            self.process_detections_and_track(frm)
            fc += 1
            if fc % 30 == 0: fps = self.calculate_fps(start, fc)
            prog = (fc / total) * 100
            x, y = 50, frm.shape[0] - 50
            cv2.rectangle(frm, (x,y), (x+400, y+20), (100,100,100), -1)
            cv2.rectangle(frm, (x,y), (x+int(400*prog/100), y+20), (0,255,0), -1)
            cv2.putText(frm, f"{prog:.1f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            self.display_counts_on_frame(frm, fps)
            cv2.imshow("Video", frm)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
        cap.release()
        cv2.destroyAllWindows()
