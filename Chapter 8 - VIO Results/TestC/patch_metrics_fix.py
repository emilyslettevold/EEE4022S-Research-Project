#!/usr/bin/env python3
"""
patch_metrics_fix.py - fix the depth counts in the per-frame VINS metrics (logging only; VINS behaviour unchanged).

Bug: n_depth_guess counted guessed-depth features over the WHOLE feature list, while n_win counts only
window features (>= 4 observations). So guessed/n_win could exceed 1, and n_depth_meas (= depths - guesses)
was clipped to 0.  Fix: count both inside the window loop, per feature.
Also adds a column n_depth_cap = window features whose depth sits at the 100 m upper bound (>= 99 m),
which flags "infinitely far" features, as seen on the rail.

Run from the VINS-Fusion repo root (after patch_vins_stereo.py and patch_vins_metrics.py), then rebuild.
"""
import re, sys
from pathlib import Path

fm = Path("vins_estimator/src/estimator/feature_manager.cpp")
em = Path("vins_estimator/src/estimator/estimator.cpp")
for f in (fm, em):
    if not f.exists():
        sys.exit(f"Not found: {f} - run from the VINS-Fusion repo root")

s = fm.read_text()
if "vinsPatchIsFallback" not in s:
    if "g_fallback_depth_ids" not in s:
        sys.exit("g_fallback_depth_ids not found - stereo patch missing")
    s += "\n// PATCH_METRICS_FIX (RAV2026-02): is this feature's depth still a guess?\n" \
         "bool vinsPatchIsFallback(int id) { return g_fallback_depth_ids.count(id) > 0; }\n"
    fm.write_text(s)
    print("feature_manager.cpp: vinsPatchIsFallback added")
else:
    print("feature_manager.cpp: helper already present")

e = em.read_text()
if "PATCH_METRICS_FIX" in e:
    sys.exit("estimator.cpp: already fixed")
if "PATCH_METRICS" not in e:
    sys.exit("estimator.cpp: patch_vins_metrics.py has not been applied")

decl = "int vinsPatchCountFallback(const std::list<FeaturePerId> &feats);"
if e.count(decl) != 1:
    sys.exit("declaration of vinsPatchCountFallback not found once")
e = e.replace(decl, decl + "\nbool vinsPatchIsFallback(int id);  // PATCH_METRICS_FIX")

pat = re.compile(r"int n_win = 0, n_meas = 0;.*?n_meas = std::max\(0, \(int\)deps\.size\(\) - n_guess\);", re.S)
m = pat.findall(e)
if len(m) != 1:
    sys.exit(f"depth-count block found {len(m)} times (expected 1) - paste the PATCH_METRICS block to Claude")
new = """int n_win = 0, n_meas = 0, n_guess = 0, n_cap = 0;   // PATCH_METRICS_FIX: counted per window feature
                std::vector<double> deps;
                for (const auto &f : f_manager.feature)
                {
                    if (f.feature_per_frame.size() < 4) continue;
                    n_win++;
                    if (vinsPatchIsFallback(f.feature_id)) n_guess++;
                    else if (f.estimated_depth > 0) n_meas++;
                    if (f.estimated_depth > 0) deps.push_back(f.estimated_depth);
                    if (f.estimated_depth >= 99.0) n_cap++;
                }"""
e = pat.sub(lambda _: new, e)

h_old = 'td_ms\\n";'
if e.count(h_old) != 1:
    sys.exit("CSV header end 'td_ms\\n\";' not found once")
e = e.replace(h_old, 'td_ms,n_depth_cap\\n";')
r_old = '<< td * 1000.0 << "\\n";'
if e.count(r_old) != 1:
    sys.exit("CSV row end not found once")
e = e.replace(r_old, '<< td * 1000.0 << "," << n_cap << "\\n";')

Path(str(em) + ".orig_metricsfix").write_text(em.read_text())
em.write_text(e)
print("estimator.cpp: depth counts fixed, n_depth_cap column added  (backup: estimator.cpp.orig_metricsfix)")
