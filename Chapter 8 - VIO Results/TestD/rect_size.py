#!/usr/bin/env python3
# rect_size.py - size of the smallest rectangle around a VINS trajectory (in its plane of motion) and its start-to-end distance.
#   python3 ~/vio_data/rect_size.py ~/vio_data/testD/square2/square2_SO.csv
import sys, numpy as np
P = np.array([[float(v) for v in l.split(",")[1:4]] for l in open(sys.argv[1]) if l[:1].isdigit()])
c = P - P.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False); Q = c @ Vt[:2].T   # plane of motion
best = min(((np.ptp(Q @ [np.cos(a), np.sin(a)]) * np.ptp(Q @ [-np.sin(a), np.cos(a)]), a) for a in np.radians(np.arange(0, 90, 0.5))))
a = best[1]; w, h = np.ptp(Q @ [np.cos(a), np.sin(a)]), np.ptp(Q @ [-np.sin(a), np.cos(a)])
print(f"{sys.argv[1]}: bounding rectangle {max(w,h):.2f} x {min(w,h):.2f} m, start-end {np.linalg.norm(P[-1]-P[0]):.3f} m")
