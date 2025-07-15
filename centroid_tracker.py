import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid, class_id):
        self.objects[self.next_object_id] = (centroid, class_id)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects, class_ids):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for (x1, y1, x2, y2) in rects])
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], class_ids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid][0] for oid in object_ids]
            D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                oid = object_ids[r]
                self.objects[oid] = (input_centroids[c], class_ids[c])
                self.disappeared[oid] = 0
                used_rows.add(r)
                used_cols.add(c)
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols
            if D.shape[0] >= D.shape[1]:
                for r in unused_rows:
                    oid = object_ids[r]
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self.deregister(oid)
            else:
                for c in unused_cols:
                    self.register(input_centroids[c], class_ids[c])
        return self.objects
