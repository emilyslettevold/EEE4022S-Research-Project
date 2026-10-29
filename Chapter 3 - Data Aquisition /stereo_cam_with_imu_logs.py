#!/usr/bin/env python3
"""
Stereo camera control for BlueROV - two StellarHD cameras, four modes.

  preview       MJPG -> H264 to MediaMTX. Nothing written to disk.
  record-mjpg   Camera's MJPG bytes straight to SSD, plus H264 preview.
  record-yuyv   Uncompressed YUYV to SSD, plus H264 preview.
  calibrate     Let auto exposure/WB settle, lock both cameras to the
                RIGHT camera's converged values, then record.

Camera controls are left on AUTO by default. They are locked in calibrate
mode, and in either record mode when --lock is given.

Recordings are never resolution-capped. Only the preview stream is, at
720p, and it is only ever downscaled - never upscaled.

Every mode that writes to disk also runs the IMU timesync logger
(imu_logger.py, same folder) and saves imu_timesync.csv into the session
folder. Use it with extract_imu.py to cut this recording's IMU data out of
the ArduSub BIN, on the same clock as arrival_monotonic_ns. Needs the
BlueOS MAVLink endpoint (UDP Client -> 127.0.0.1:14670). --no-imu skips it.

Requires MediaMTX listening on :8555 (RTSP in) and :8889 (WebRTC out).

Usage:
    ./stereo_cam.py --list-modes
    ./stereo_cam.py --mode preview --cap 1280x720 --fps 30 --preview-fps 15
    ./stereo_cam.py --mode record-mjpg --cap 1600x1200 --fps 30 --lock
    ./stereo_cam.py --mode record-yuyv --cap 1600x1200 --fps 5
    ./stereo_cam.py --mode calibrate --format mjpg --cap 1600x1200 --fps 15

    for d in /dev/cam_left /dev/cam_right; do   v4l2-ctl -d $d --set-ctrl=auto_exposure=1;   sleep 0.5;   echo "$d: $(v4l2-ctl -d $d --get-ctrl=exposure_time_absolute)";   v4l2-ctl -d $d --set-ctrl=auto_exposure=3; done
    
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402

# IMU timesync logger - optional, so a missing pymavlink never blocks a
# recording. It just prints a warning instead.
try:
    from imu_logger import ImuLogger
except ImportError:
    ImuLogger = None

# ---- devices ----------------------------------------------------------
LEFT_DEV = "/dev/cam_left"
RIGHT_DEV = "/dev/cam_right"

# ---- preview stream ---------------------------------------------------
# Hard ceiling on the preview. Recordings are never capped by this.
PREVIEW_MAX_W, PREVIEW_MAX_H = 1280, 720
PREVIEW_BITRATE = 2500                     # kbps per camera
DEFAULT_PREVIEW_FPS = 10

# ---- defaults ---------------------------------------------------------
DEFAULT_CAP = (1280, 720)
DEFAULT_FPS = 15
DECODER = "jpegdec"

# ---- output -----------------------------------------------------------
RTSP_BASE = "rtsp://127.0.0.1:8555"
SSD_ROOT = "/media/ssd"
SD_ROOT = str(Path.home() / "captures")    # staging area on the SD card
CALIB_FILE = Path(SSD_ROOT) / "camera_calibration.json"

# ---- IMU timesync -----------------------------------------------------
# BlueOS MAVLink endpoint: UDP Client -> 127.0.0.1:14670
MAVLINK_CONN = "udpin:0.0.0.0:14670"
IMU_READY_TIMEOUT_S = 5                    # wait for first timesync reply

# ---- camera controls --------------------------------------------------
# Controls whose auto-converged values we capture during calibration and
# then pin to both cameras.
LOCKED_CONTROLS = [
    "exposure_time_absolute",
    "white_balance_temperature",
    "gain",
    "brightness",
    "contrast",
    "saturation",
    "hue",
    "gamma",
    "sharpness",
    "backlight_compensation",
]

# Applied regardless of auto/manual - never wanted underwater.
STATIC_CONTROLS = [
    ("power_line_frequency", 0),           # no mains lighting underwater
    ("exposure_dynamic_framerate", 0),     # stop silent fps drops
]

AUTO_SETTLE_S = 5                          # calibrate mode settling time
# -----------------------------------------------------------------------


# ======================================================================
#  v4l2 helpers
# ======================================================================

def v4l2(dev, *args):
    return subprocess.run(["v4l2-ctl", "-d", dev, *args],
                          capture_output=True, text=True)


def set_ctrl(dev, name, value):
    return v4l2(dev, f"--set-ctrl={name}={value}").returncode == 0


def read_ctrls(dev):
    """Return {control_name: int_value} for everything the camera reports."""
    r = v4l2(dev, "--list-ctrls")
    if r.returncode != 0:
        sys.exit(f"Cannot read controls from {dev}: {r.stderr.strip()}")
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\(\w+\)", line)
        if not m:
            continue
        v = re.search(r"value=(-?\d+)", line)
        if v:
            out[m.group(1)] = int(v.group(1))
    return out


def list_modes(dev):
    """Parse --list-formats-ext into {fourcc: {(w,h): [fps,...]}}."""
    r = v4l2(dev, "--list-formats-ext")
    if r.returncode != 0:
        sys.exit(f"Cannot list modes for {dev}: {r.stderr.strip()}")
    modes, fourcc, size = {}, None, None
    for line in r.stdout.splitlines():
        m = re.search(r"\]:\s*'(\w+)'", line)
        if m:
            fourcc = m.group(1)
            modes[fourcc] = {}
            continue
        m = re.search(r"Size:\s*Discrete\s*(\d+)x(\d+)", line)
        if m and fourcc:
            size = (int(m.group(1)), int(m.group(2)))
            modes[fourcc][size] = []
            continue
        m = re.search(r"\(([\d.]+)\s*fps\)", line)
        if m and fourcc and size:
            modes[fourcc][size].append(round(float(m.group(1))))
    return modes


def print_modes():
    for dev in (LEFT_DEV, RIGHT_DEV):
        print(f"\n{dev}")
        for fourcc, sizes in list_modes(dev).items():
            print(f"  {fourcc}")
            for (w, h), fps in sorted(sizes.items(), reverse=True):
                rates = ", ".join(str(f) for f in sorted(fps))
                print(f"    {f'{w}x{h}':<12} {rates}")


def validate_mode(dev, fourcc, w, h, fps):
    modes = list_modes(dev)
    if fourcc not in modes:
        sys.exit(f"{dev} does not support {fourcc}. "
                 f"Available: {', '.join(modes)}")
    if (w, h) not in modes[fourcc]:
        avail = ", ".join(f"{a}x{b}"
                          for a, b in sorted(modes[fourcc], reverse=True))
        sys.exit(f"{dev} does not support {fourcc} at {w}x{h}.\n"
                 f"Available: {avail}")
    if fps not in modes[fourcc][(w, h)]:
        avail = ", ".join(str(f) for f in sorted(modes[fourcc][(w, h)]))
        sys.exit(f"{dev} does not support {fourcc} {w}x{h} at {fps}fps.\n"
                 f"Available: {avail}")


# ======================================================================
#  control locking
# ======================================================================

def apply_static_controls():
    for dev in (LEFT_DEV, RIGHT_DEV):
        for name, value in STATIC_CONTROLS:
            set_ctrl(dev, name, value)


def set_auto(dev, on):
    """Enable or disable auto exposure and auto white balance."""
    set_ctrl(dev, "auto_exposure", 3 if on else 1)
    set_ctrl(dev, "white_balance_automatic", 1 if on else 0)


def lock_to_values(values, verbose=True):
    """Pin both cameras to an identical set of control values.

    The auto switches must go off FIRST - exposure_time_absolute and
    white_balance_temperature read as inactive while auto is on, and
    writes to them are silently discarded.
    """
    for dev in (LEFT_DEV, RIGHT_DEV):
        set_auto(dev, False)

    rejected = {LEFT_DEV: [], RIGHT_DEV: []}
    for dev in (LEFT_DEV, RIGHT_DEV):
        for name, value in values.items():
            if not set_ctrl(dev, name, value):
                rejected[dev].append(name)

    if verbose:
        if values:
            print("\n  Locked both cameras to:")
            for name, value in values.items():
                print(f"    {name:<32} {value}")
        else:
            print("\n  Auto switched off; no values written.")
            print("  Cameras hold their own converged settings.")
        for dev in (LEFT_DEV, RIGHT_DEV):
            if rejected[dev]:
                print(f"    {dev} rejected: {', '.join(rejected[dev])}")
    return rejected


def calibrate_and_lock(exposure=None, wb=None):
    """Let auto settle, then pin BOTH cameras to the RIGHT camera's values.

    ORDER MATTERS. While auto is enabled, exposure_time_absolute and
    white_balance_temperature are marked inactive and report stale
    factory defaults, not what auto actually converged on. UVC cameras
    latch the live value into the manual control at the moment auto is
    switched off - so we must disable auto FIRST, then read.

    Reading before disabling auto is why an earlier version locked 4600K
    regardless of the actual lighting.
    """
    print(f"\n  Auto exposure/WB settling for {AUTO_SETTLE_S}s...")
    for dev in (LEFT_DEV, RIGHT_DEV):
        set_auto(dev, True)
    time.sleep(AUTO_SETTLE_S)

    stale = {d: read_ctrls(d) for d in (LEFT_DEV, RIGHT_DEV)}

    # switching auto off latches whatever auto converged on
    print("  Disabling auto to latch the converged values...")
    for dev in (LEFT_DEV, RIGHT_DEV):
        set_auto(dev, False)
    time.sleep(0.5)

    before = {d: read_ctrls(d) for d in (LEFT_DEV, RIGHT_DEV)}

    print("\n  Converged values (read after disabling auto):")
    print(f"    {'control':<32}{'left':<12}{'right':<12}{'was':<12}")
    for name in LOCKED_CONTROLS:
        lv = before[LEFT_DEV].get(name, "-")
        rv = before[RIGHT_DEV].get(name, "-")
        was = stale[RIGHT_DEV].get(name, "-")
        note = ""
        if lv != rv:
            note = "  differ"
        print(f"    {name:<32}{str(lv):<12}{str(rv):<12}{str(was):<12}{note}")

    if all(before[RIGHT_DEV].get(n) == stale[RIGHT_DEV].get(n)
           for n in ("exposure_time_absolute", "white_balance_temperature")):
        print("\n  WARNING: values did not change after disabling auto.\n"
              "  This camera may not report converged values at all.\n"
              "  Set them by hand with --exposure and --wb instead.")

    values = {n: before[RIGHT_DEV][n] for n in LOCKED_CONTROLS
              if n in before[RIGHT_DEV]}

    # If a control reads the same before and after switching auto off, the
    # camera never reported what auto converged on - it is a stale default.
    # Writing it would OVERWRITE the good value the camera is now holding
    # internally, which is what made everything come out yellow. Drop it
    # and let the camera keep its own latched value.
    untrusted = []
    for name in ("exposure_time_absolute", "white_balance_temperature"):
        if name in values and before[RIGHT_DEV].get(name) == \
                stale[RIGHT_DEV].get(name):
            untrusted.append(name)
            values.pop(name)

    if untrusted:
        print(f"\n  These read as stale defaults, so they will NOT be "
              f"written:\n    {', '.join(untrusted)}")
        print("  The cameras keep whatever auto converged on internally.")
        print("  Downside: left and right may differ slightly. Use "
              "--exposure / --wb\n  to force identical values if that "
              "matters for your stereo work.")

    # explicit overrides win over anything auto worked out
    if exposure is not None:
        values["exposure_time_absolute"] = exposure
        print(f"\n  Overriding exposure -> {exposure}")
    if wb is not None:
        values["white_balance_temperature"] = wb
        print(f"  Overriding white balance -> {wb}K")

    print("\n  Adopting the RIGHT camera's values for both cameras.")
    lock_to_values(values)

    after = {d: read_ctrls(d) for d in (LEFT_DEV, RIGHT_DEV)}
    print("\n  Verified after locking:")
    print(f"    {'control':<32}{'left':<12}{'right':<12}")
    bad = False
    for name in values:
        lv = after[LEFT_DEV].get(name, "-")
        rv = after[RIGHT_DEV].get(name, "-")
        ok = (lv == rv == values[name])
        bad = bad or not ok
        print(f"    {name:<32}{str(lv):<12}{str(rv):<12}"
              f"{'' if ok else '  <-- DID NOT TAKE'}")
    if bad:
        print("\n  WARNING: some controls did not hold. The camera may be\n"
              "  refusing writes while streaming.")

    target = CALIB_FILE
    if not Path(SSD_ROOT).is_mount():
        target = Path(SD_ROOT) / "camera_calibration.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(
            {"saved": time.strftime("%Y-%m-%d %H:%M:%S"),
             "values": values}, indent=2))
        print(f"\n  Saved to {target}")
    except OSError as e:
        print(f"\n  Could not save calibration: {e}")
    return values


def load_saved_lock():
    for candidate in (CALIB_FILE, Path(SD_ROOT) / "camera_calibration.json"):
        if candidate.exists():
            data = json.loads(candidate.read_text())
            print(f"\n  Loading locked values from {candidate} "
                  f"(saved {data.get('saved', '?')})")
            return data["values"]
    sys.exit(f"No saved calibration found.\n"
             f"Run --mode calibrate first, or drop --lock.")


# ======================================================================
#  pipeline construction
# ======================================================================

class TimestampLog:
    """Buffers frame timestamps in RAM, flushes to CSV at the end."""

    def __init__(self, name):
        self.name = name
        self.rows = []

    def add(self, i, pts_ns, arrival_ns):
        self.rows.append((i, pts_ns, arrival_ns))

    def write(self, path, wall_epoch, base_ns):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "pts_ns", "arrival_monotonic_ns",
                        "wall_clock_epoch_at_start", "base_time_ns"])
            for row in self.rows:
                w.writerow([*row, wall_epoch, base_ns])
        print(f"  {self.name}: {len(self.rows)} frames -> {path}")


def make_factory(pipeline, suffix):
    def mk(factory, name):
        el = Gst.ElementFactory.make(factory, f"{name}_{suffix}")
        if el is None:
            sys.exit(f"Missing GStreamer element: {factory}\n"
                     f"  rtspclientsink: sudo apt install gstreamer1.0-rtsp\n"
                     f"  x264enc:        sudo apt install "
                     f"gstreamer1.0-plugins-ugly")
        pipeline.add(el)
        return el
    return mk


def link_chain(chain):
    for a, b in zip(chain, chain[1:]):
        if not a.link(b):
            sys.exit(f"Failed to link {a.get_name()} -> {b.get_name()}")


def preview_size(cap_w, cap_h):
    """Fit within the preview ceiling, preserving aspect. Never upscales."""
    if cap_w <= PREVIEW_MAX_W and cap_h <= PREVIEW_MAX_H:
        return cap_w, cap_h
    s = min(PREVIEW_MAX_W / cap_w, PREVIEW_MAX_H / cap_h)
    # H264 wants even dimensions
    return (int(cap_w * s) // 2) * 2, (int(cap_h * s) // 2) * 2


def attach_probe(element, log):
    counter = {"n": 0}

    def probe(pad, info):
        buf = info.get_buffer()
        if buf is not None:
            log.add(counter["n"], buf.pts, time.monotonic_ns())
            counter["n"] += 1
        return Gst.PadProbeReturn.OK

    element.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe)


def build_branch(pipeline, device, log, fourcc, w, h, fps,
                 preview_fps, save_to):
    """One camera: source -> tee -> (raw to disk) + (H264 preview).

    Frame rate reduction happens BEFORE decoding, so dropped frames cost
    nothing to decode. That is the main CPU saving in this pipeline.
    """
    mk = make_factory(pipeline, log.name)
    is_jpeg = fourcc == "MJPG"

    src = mk("v4l2src", "src")
    src.set_property("device", device)
    src.set_property("io-mode", 2)
    src.set_property("do-timestamp", True)

    caps = mk("capsfilter", "caps")
    caps.set_property("caps", Gst.Caps.from_string(
        f"image/jpeg,width={w},height={h},framerate={fps}/1" if is_jpeg else
        f"video/x-raw,format=YUY2,width={w},height={h},framerate={fps}/1"))

    src.link(caps)
    attach_probe(caps, log)

    tee = mk("tee", "tee")
    caps.link(tee)

    # ---- disk branch: exactly what the camera produced, unmodified ----
    if save_to is not None:
        q_rec = mk("queue", "q_rec")
        q_rec.set_property("max-size-buffers", 8)
        q_rec.set_property("leaky", 0)        # never drop recorded frames
        mux = mk("matroskamux", "mux")
        fsink = mk("filesink", "fsink")
        fsink.set_property("location", str(save_to))
        fsink.set_property("sync", False)
        tee.link(q_rec)
        if is_jpeg:
            link_chain([q_rec, mk("jpegparse", "jparse"), mux, fsink])
        else:
            link_chain([q_rec, mux, fsink])

    # ---- preview branch ----
    q_prev = mk("queue", "q_prev")
    q_prev.set_property("leaky", 2)           # drop preview under load
    q_prev.set_property("max-size-buffers", 2)
    tee.link(q_prev)
    chain = [q_prev]

    # decimate first, while frames are still cheap to discard
    if preview_fps and preview_fps < fps:
        rate = mk("videorate", "rate")
        rate.set_property("drop-only", True)  # never duplicate frames
        rcaps = mk("capsfilter", "rcaps")
        rcaps.set_property("caps", Gst.Caps.from_string(
            f"image/jpeg,framerate={preview_fps}/1" if is_jpeg else
            f"video/x-raw,framerate={preview_fps}/1"))
        chain += [rate, rcaps]
        out_fps = preview_fps
    else:
        out_fps = fps

    if is_jpeg:
        chain.append(mk(DECODER, "dec"))
    chain.append(mk("videoconvert", "conv"))

    pw, ph = preview_size(w, h)
    if (pw, ph) != (w, h):
        scale = mk("videoscale", "scale")
        scale.set_property("method", 0)       # nearest: cheapest
        pcaps = mk("capsfilter", "pcaps")
        pcaps.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw,width={pw},height={ph}"))
        chain += [scale, pcaps]

    enc = mk("x264enc", "enc")
    enc.set_property("speed-preset", 1)       # ultrafast
    enc.set_property("tune", 4)               # zerolatency
    enc.set_property("bitrate", PREVIEW_BITRATE)
    enc.set_property("bframes", 0)            # WebRTC cannot handle these
    enc.set_property("key-int-max", max(out_fps, 1))
    rtsp = mk("rtspclientsink", "rtsp")
    rtsp.set_property("location", f"{RTSP_BASE}/{log.name}")
    rtsp.set_property("protocols", 4)         # TCP
    chain += [enc, mk("h264parse", "parse"), rtsp]

    link_chain(chain)
    return pw, ph, out_fps


# ======================================================================
#  IMU timesync
# ======================================================================

def start_imu_logger(outdir):
    """Start the timesync logger and wait for its first reply.

    Started BEFORE the cameras and stopped AFTER them, so the timesync
    window always brackets the whole recording. Never blocks a recording:
    if anything is missing it warns and returns None.
    """
    if ImuLogger is None:
        print("  WARNING: imu_logger.py / pymavlink not importable - "
              "recording WITHOUT IMU timesync")
        return None
    imu = ImuLogger(MAVLINK_CONN, out_prefix=str(outdir / "imu"),
                    stream_imu=False)
    imu.start()
    deadline = time.monotonic() + IMU_READY_TIMEOUT_S
    while imu.n_sync == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    if imu.n_sync:
        print("  IMU timesync running")
    else:
        print("  WARNING: no timesync reply from the autopilot yet - "
              "recording anyway.\n  Check the BlueOS MAVLink endpoint "
              "(UDP Client -> 127.0.0.1:14670).")
    return imu


# ======================================================================
#  output directory
# ======================================================================

def prepare_outdir(args, need_bytes):
    """Pick and validate the capture directory.

    --storage ssd  (default) writes straight to the SSD.
    --storage sd   stages on the SD card, then offers to copy afterwards.
                   Use this when the SSD is dropping off the bus mid-write:
                   the capture completes regardless of what the SSD does.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    staging = args.storage == "sd"
    root = Path(SD_ROOT) if staging else Path(SSD_ROOT)
    outdir = Path(args.outdir) if args.outdir else root / "sessions" / stamp

    root_dev = os.stat("/").st_dev
    probe = outdir
    while not probe.exists():
        probe = probe.parent
    on_sdcard = os.stat(probe).st_dev == root_dev

    if on_sdcard and not staging and not args.allow_sdcard:
        sys.exit(f"Refusing to write to {outdir} - that is the SD card, "
                 f"not the SSD.\nIs {SSD_ROOT} mounted?  Check: lsblk -f\n"
                 f"Use --storage sd to stage there deliberately, or "
                 f"--allow-sdcard to override.")

    try:
        outdir.mkdir(parents=True, exist_ok=True)
        t = outdir / ".writetest"
        t.write_bytes(b"x")
        t.unlink()
    except OSError as e:
        sys.exit(f"Cannot write to {outdir}: {e}\n"
                 f"exFAT ignores chown - remount with:\n"
                 f"  sudo mount -o uid=$(id -u),gid=$(id -g),noatime "
                 f"UUID=6474-DBFC {SSD_ROOT}")

    free = shutil.disk_usage(probe).free
    if need_bytes and free < need_bytes * 1.1:
        sys.exit(f"Not enough space: need ~{need_bytes / 1e9:.1f} GB, "
                 f"have {free / 1e9:.1f} GB")

    if staging:
        print(f"  STAGING on the SD card ({free / 1e9:.1f} GB free)")
        if need_bytes:
            print(f"  this capture needs ~{need_bytes / 1e9:.1f} GB")
    return outdir


def offer_transfer(outdir):
    """After a staged capture, offer to move it onto the SSD.

    Kept separate from the capture so a failing SSD cannot corrupt the
    recording - by this point the files are already safe on the SD card.
    """
    files = sorted(p for p in outdir.iterdir() if p.is_file())
    if not files:
        return
    total = sum(p.stat().st_size for p in files)

    print(f"\n  Captured {len(files)} files, {total / 1e9:.2f} GB, "
          f"in {outdir}")

    ssd = Path(SSD_ROOT)
    if not ssd.is_mount():
        print(f"  {SSD_ROOT} is not mounted - leaving files on the SD card.")
        print(f"  Mount it and copy manually when ready:")
        print(f"    rsync -avP {outdir}/ {SSD_ROOT}/sessions/{outdir.name}/")
        return

    free = shutil.disk_usage(ssd).free
    if free < total * 1.05:
        print(f"  SSD has only {free / 1e9:.1f} GB free - not copying.")
        return

    try:
        answer = input(f"\n  Copy to {SSD_ROOT}? [y/N] ").strip().lower()
    except EOFError:
        answer = "n"
    if answer not in ("y", "yes"):
        print(f"  Left on the SD card. Copy later with:")
        print(f"    rsync -avP {outdir}/ {SSD_ROOT}/sessions/{outdir.name}/")
        return

    dest = ssd / "sessions" / outdir.name
    dest.mkdir(parents=True, exist_ok=True)

    # rsync when available: resumable, verifies as it goes, and survives
    # the SSD dropping mid-copy without losing the source files
    if shutil.which("rsync"):
        r = subprocess.run(["rsync", "-avP", f"{outdir}/", f"{dest}/"])
        ok = r.returncode == 0
    else:
        ok = True
        for p in files:
            try:
                shutil.copy2(p, dest / p.name)
                print(f"    copied {p.name}")
            except OSError as e:
                print(f"    FAILED {p.name}: {e}")
                ok = False

    if not ok:
        print(f"\n  Copy failed. Originals are intact in {outdir}")
        print(f"  Retry with: rsync -avP {outdir}/ {dest}/")
        return

    # verify sizes before offering to delete anything
    mismatched = [p.name for p in files
                  if not (dest / p.name).exists()
                  or (dest / p.name).stat().st_size != p.stat().st_size]
    if mismatched:
        print(f"\n  Size mismatch on: {', '.join(mismatched)}")
        print(f"  Keeping the SD card copies.")
        return

    print(f"\n  Copied and verified to {dest}")
    try:
        if input("  Delete the SD card copies? [y/N] ").strip().lower() \
                in ("y", "yes"):
            for p in files:
                p.unlink()
            outdir.rmdir()
            print("  SD card copies removed.")
    except EOFError:
        pass


# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="preview",
                    choices=["preview", "record-mjpg", "record-yuyv",
                             "calibrate"])
    ap.add_argument("--cap", default=None, metavar="WxH",
                    help="capture resolution (see --list-modes)")
    ap.add_argument("--fps", type=int, default=None,
                    help="capture framerate (must be natively supported)")
    ap.add_argument("--preview-fps", type=int, default=None,
                    help=f"preview framerate, default {DEFAULT_PREVIEW_FPS}; "
                         f"frames are dropped before decoding")
    ap.add_argument("--format", default="mjpg", choices=["mjpg", "yuyv"],
                    help="calibrate mode: which format to record in")
    ap.add_argument("--lock", action="store_true",
                    help="record modes: lock controls from the last "
                         "calibration instead of leaving them on auto")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--exposure", type=int, default=None,
                    help="manual exposure, units of 100us (1-5000). "
                         "Overrides whatever auto converged on.")
    ap.add_argument("--wb", type=int, default=None,
                    help="manual white balance in K (2800-6500). "
                         "Lower is warmer. Overrides auto.")
    ap.add_argument("--storage", default="ssd", choices=["ssd", "sd"],
                    help="ssd (default) writes straight to the SSD; "
                         "sd stages on the SD card then offers to copy")
    ap.add_argument("--allow-sdcard", action="store_true")
    ap.add_argument("--decoder", default="jpegdec",
                    choices=["jpegdec", "avdec_mjpeg"])
    ap.add_argument("--no-imu", action="store_true",
                    help="do not run the IMU timesync logger")
    ap.add_argument("--list-modes", action="store_true",
                    help="print what both cameras support, then exit")
    args = ap.parse_args()

    if args.list_modes:
        print_modes()
        return

    global DECODER
    DECODER = args.decoder

    if args.mode == "record-yuyv":
        fourcc = "YUYV"
    elif args.mode == "calibrate":
        fourcc = "YUYV" if args.format == "yuyv" else "MJPG"
    else:
        fourcc = "MJPG"

    w, h = (tuple(int(v) for v in args.cap.lower().split("x"))
            if args.cap else DEFAULT_CAP)
    fps = args.fps if args.fps else (5 if fourcc == "YUYV" else DEFAULT_FPS)
    preview_fps = (args.preview_fps if args.preview_fps
                   else min(DEFAULT_PREVIEW_FPS, fps))

    for dev in (LEFT_DEV, RIGHT_DEV):
        validate_mode(dev, fourcc, w, h, fps)

    writes_disk = args.mode != "preview"
    need = 0
    if writes_disk and args.duration:
        if fourcc == "YUYV":
            need = 2 * w * h * 2 * fps * args.duration
        else:
            # MJPG is roughly 10:1 on typical scenes; a rough guide is
            # enough here, it only feeds the free-space warning
            need = int(2 * w * h * 2 * fps * args.duration * 0.1)
    outdir = prepare_outdir(args, need) if writes_disk else None

    lsave = outdir / "left.mkv" if writes_disk else None
    rsave = outdir / "right.mkv" if writes_disk else None

    Gst.init(None)
    pipeline = Gst.Pipeline.new("stereo_cam")
    left_log, right_log = TimestampLog("left"), TimestampLog("right")

    pw, ph, out_fps = build_branch(pipeline, LEFT_DEV, left_log, fourcc,
                                   w, h, fps, preview_fps, lsave)
    build_branch(pipeline, RIGHT_DEV, right_log, fourcc,
                 w, h, fps, preview_fps, rsave)

    loop = GLib.MainLoop()

    def on_message(bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"ERROR: {err}\n{debug}", file=sys.stderr)
            loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            loop.quit()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)

    # IMU timesync starts before the cameras so its window covers frame 0
    imu = None
    if writes_disk and not args.no_imu:
        imu = start_imu_logger(outdir)

    wall_epoch = time.time()
    pipeline.set_state(Gst.State.PLAYING)
    pipeline.get_state(Gst.CLOCK_TIME_NONE)
    base_ns = pipeline.get_base_time()

    print(f"\n{args.mode.upper()}: {fourcc} {w}x{h} @ {fps}fps")
    if writes_disk:
        print(f"  recording to {outdir}  (uncapped, as captured)")
    print(f"  preview: H264 {pw}x{ph} @ {out_fps}fps")
    print(f"  streams: {RTSP_BASE}/left  and  {RTSP_BASE}/right")
    print(f"  view at http://<pi-ip>:8889/left")

    apply_static_controls()

    def do_controls():
        if args.mode == "calibrate":
            calibrate_and_lock(exposure=args.exposure, wb=args.wb)
        elif args.lock:
            values = load_saved_lock()
            if args.exposure is not None:
                values["exposure_time_absolute"] = args.exposure
            if args.wb is not None:
                values["white_balance_temperature"] = args.wb
            lock_to_values(values)
        elif args.exposure is not None or args.wb is not None:
            # manual values without a saved calibration
            values = {}
            if args.exposure is not None:
                values["exposure_time_absolute"] = args.exposure
            if args.wb is not None:
                values["white_balance_temperature"] = args.wb
            lock_to_values(values)
        else:
            for dev in (LEFT_DEV, RIGHT_DEV):
                set_auto(dev, True)
            print("\n  Camera controls left on AUTO")
        sys.stdout.flush()
        return False

    # controls must be applied after the device is open and streaming;
    # opening a UVC device resets them
    GLib.timeout_add_seconds(1, do_controls)

    def stop(*_):
        print("\nStopping...")
        pipeline.send_event(Gst.Event.new_eos())
        return False

    signal.signal(signal.SIGINT, lambda *_: stop())
    if args.duration:
        total = args.duration + (AUTO_SETTLE_S + 2
                                 if args.mode == "calibrate" else 0)
        GLib.timeout_add_seconds(int(total), stop)

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

        # IMU timesync stops after the cameras so its window covers the
        # last frame
        if imu is not None:
            imu.stop()
            print(f"  IMU timesync: {imu.n_sync} pairs -> "
                  f"{outdir / 'imu_timesync.csv'}")
            if imu.n_sync == 0:
                print("  WARNING: no timesync pairs recorded - this "
                      "session's IMU window cannot be found automatically.")

        if outdir:
            print("Writing timestamp logs:")
            left_log.write(outdir / "left_timestamps.csv", wall_epoch, base_ns)
            right_log.write(outdir / "right_timestamps.csv",
                            wall_epoch, base_ns)

        n = min(len(left_log.rows), len(right_log.rows))
        if n:
            d = sorted(abs(left_log.rows[i][1] - right_log.rows[i][1]) / 1e6
                       for i in range(n))
            print(f"\nPair offset over {n} frames (ms): "
                  f"median {d[n // 2]:.1f}, worst {d[-1]:.1f}")

        if outdir and args.storage == "sd":
            offer_transfer(outdir)


if __name__ == "__main__":
    main()
