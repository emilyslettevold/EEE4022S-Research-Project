#!/usr/bin/env python3
"""
patch_fastinit.py - make ORB-SLAM3 v1.0 read IMU.fastInit from v1.0-format ("File.version: 1.0") configs.

Stock v1.0 reads IMU.fastInit only in ParseIMUParamFile(), which is used for OLD-format configs.
With a v1.0-format config the constructor calls newParameterLoader(settings) instead, so the
setting is ignored and the 0.5 m/s^2 acceleration check (Tracking.cc, "!mFastInit && ...") always applies.

This adds, right after newParameterLoader(settings); in the Tracking constructor, a direct read of
IMU.fastInit from the same settings file. Absent or 0 -> false, so configs WITHOUT the key behave
exactly as before (the default runs stay valid).

Usage (Mac):  python3 patch_fastinit.py Tracking.cc      (writes Tracking.cc.orig_fastinit backup)
"""
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "Tracking.cc")
s = p.read_text()
if "PATCH_FASTINIT" in s:
    sys.exit("already patched - nothing to do")

call = "newParameterLoader(settings);"
n = s.count(call)
if n != 1:
    sys.exit(f"expected '{call}' exactly once, found {n} - paste: grep -n newParameterLoader Tracking.cc")
if "strSettingPath" not in s:
    sys.exit("constructor argument strSettingPath not found - paste the Tracking::Tracking( line")

add = call + """
        {   // PATCH_FASTINIT (RAV2026-02): honour IMU.fastInit with v1.0-format configs
            cv::FileStorage fsFastInit(strSettingPath.c_str(), cv::FileStorage::READ);
            cv::FileNode nFastInit = fsFastInit["IMU.fastInit"];
            mFastInit = (!nFastInit.empty() && static_cast<int>(nFastInit) != 0);
            std::cout << "IMU.fastInit (patched reader): " << (mFastInit ? 1 : 0) << std::endl;
        }"""
Path(str(p) + ".orig_fastinit").write_text(s)
p.write_text(s.replace(call, add))
print(f"patched {p}  (backup: {p}.orig_fastinit)")
