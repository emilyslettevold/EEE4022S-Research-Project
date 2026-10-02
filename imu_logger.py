"""
imu_logger.py - run alongside stereo_cam.py on the Pi.

Logs, on the SAME clock as the camera timestamps (time.monotonic_ns):
  <prefix>_timesync.csv : Pi<->flight-controller clock pairs (for cropping/aligning the BIN)
  <prefix>_rawimu.csv   : RAW_IMU samples streamed over MAVLink (optional, lower rate than BIN)

Usage inside stereo_cam.py:
    from imu_logger import ImuLogger
    imu = ImuLogger("udpin:0.0.0.0:14660", out_prefix=session_name)
    imu.start()
    ...start camera recording...
    ...stop camera recording...
    imu.stop()

Standalone rate test (10 s):
    python imu_logger.py udpin:0.0.0.0:14660 test_run
"""
import csv
import sys
import threading
import time

from pymavlink import mavutil

MAV_CMD_SET_MESSAGE_INTERVAL = 511
RAW_IMU_ID = 27
AUTOPILOT_SYS, AUTOPILOT_COMP = 1, 1  # ArduSub defaults


class ImuLogger:
    def __init__(self, conn_str, out_prefix, imu_hz=200, sync_hz=2.0, stream_imu=True):
        self.conn_str = conn_str
        self.out_prefix = out_prefix
        self.imu_hz = imu_hz
        self.sync_period_ns = int(1e9 / sync_hz)
        self.stream_imu = stream_imu
        self._stop = threading.Event()
        self._thread = None
        self.n_imu = 0
        self.n_sync = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _set_interval(self, m, msg_id, interval_us):
        m.mav.command_long_send(AUTOPILOT_SYS, AUTOPILOT_COMP,
                                MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                msg_id, interval_us, 0, 0, 0, 0, 0)

    def _run(self):
        m = mavutil.mavlink_connection(
            self.conn_str, source_system=255,
            source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER)
        m.wait_heartbeat(timeout=10)

        if self.stream_imu:
            self._set_interval(m, RAW_IMU_ID, int(1e6 / self.imu_hz))

        f_sync = open(f"{self.out_prefix}_timesync.csv", "w", newline="")
        w_sync = csv.writer(f_sync)
        w_sync.writerow(["pi_send_ns", "pi_recv_ns", "fc_ns"])

        f_imu = w_imu = None
        if self.stream_imu:
            f_imu = open(f"{self.out_prefix}_rawimu.csv", "w", newline="")
            w_imu = csv.writer(f_imu)
            # ArduPilot RAW_IMU: acc in milli-g, gyro in mrad/s, time_usec = FC boot time
            w_imu.writerow(["pi_recv_ns", "fc_time_usec",
                            "xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro"])

        pending = {}  # ts1 we sent -> send time
        next_sync = 0
        try:
            while not self._stop.is_set():
                now = time.monotonic_ns()
                if now >= next_sync:
                    m.mav.timesync_send(0, now)
                    pending[now] = now
                    next_sync = now + self.sync_period_ns
                    if len(pending) > 50:  # drop stale unanswered pings
                        pending.pop(next(iter(pending)))

                msg = m.recv_match(type=["RAW_IMU", "TIMESYNC"],
                                   blocking=True, timeout=0.05)
                if msg is None:
                    continue
                t = time.monotonic_ns()

                if msg.get_type() == "TIMESYNC":
                    # Replies have tc1 != 0 and echo our ts1
                    if msg.tc1 != 0 and msg.ts1 in pending:
                        w_sync.writerow([pending.pop(msg.ts1), t, msg.tc1])
                        self.n_sync += 1
                elif w_imu is not None and msg.get_srcSystem() == AUTOPILOT_SYS:
                    w_imu.writerow([t, msg.time_usec, msg.xacc, msg.yacc, msg.zacc,
                                    msg.xgyro, msg.ygyro, msg.zgyro])
                    self.n_imu += 1
        finally:
            if self.stream_imu:
                self._set_interval(m, RAW_IMU_ID, 0)  # 0 = restore default rate
            f_sync.close()
            if f_imu:
                f_imu.close()
            m.close()


def fc_to_pi_offset(timesync_csv, best_frac=0.2):
    """Return (offset_ns, fc_start_ns, fc_end_ns).
    pi_ns = fc_ns - offset_ns. Uses the lowest-RTT exchanges for the offset.
    BIN TimeUS * 1000 = fc_ns, so crop BIN rows to [fc_start_ns, fc_end_ns]."""
    rows = []
    with open(timesync_csv) as f:
        for r in csv.DictReader(f):
            s, rcv, fc = int(r["pi_send_ns"]), int(r["pi_recv_ns"]), int(r["fc_ns"])
            rows.append((rcv - s, fc - (s + rcv) // 2, fc))
    rows.sort()
    best = rows[:max(1, int(len(rows) * best_frac))]
    offsets = sorted(o for _, o, _ in best)
    fcs = [fc for _, _, fc in rows]
    return offsets[len(offsets) // 2], min(fcs), max(fcs)


if __name__ == "__main__":
    conn, prefix = sys.argv[1], sys.argv[2]
    log = ImuLogger(conn, prefix)
    log.start()
    time.sleep(10)
    log.stop()
    print(f"RAW_IMU: {log.n_imu / 10:.1f} Hz achieved, timesync pairs: {log.n_sync}")
    if log.n_sync:
        off, a, b = fc_to_pi_offset(f"{prefix}_timesync.csv")
        print(f"offset {off} ns, BIN window TimeUS {a // 1000} -> {b // 1000}")
