"""
Navis Tracker — Enhanced Vision & Control Loop
-----------------------------------------------
Key improvements over v1:
  1. REAL distance via shoulder width (pose keypoints + pinhole model).
  2. Zero-lag frame grabber: drains the MJPEG buffer every cycle so
     latest_frame is always <50 ms fresh.
  3. Stable human following:
       • Pick the person with the largest bounding-box area (closest).
       • Track shoulder MID-POINT instead of raw bbox centre.
       • Exponential moving average (EMA) smoothing on (cx, distance).
       • PID-based turning replaces simple bang-bang control.
  4. Matplotlib telemetry window (for diagnostics / calibration).
"""

import atexit
import collections
import math
import os
import signal
import threading
import time

import cv2
import matplotlib
matplotlib.use("TkAgg")   # non-blocking backend; change to 'Qt5Agg' if preferred
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, Response
from ultralytics import YOLO

load_dotenv('env')

PHONE_IP = os.getenv("PHONE_IP", "")
PICO_IP  = os.getenv("PICO_IP", "")

if not PHONE_IP or not PICO_IP:
    print("Error: PHONE_IP and PICO_IP must be set in the 'env' file.")
    exit(1)

# ──────────────────────────────────────────────────────────────────────────────
#  SPEED CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
FORWARD_SPEED  = 100
BACKWARD_SPEED = 90
MAX_TURN_SPEED = 90         # maximum turning speed (clipped by PID)
MIN_TURN_SPEED = 45         # minimum meaningful turn (overcomes static friction)

# PID gains for yaw control (tune Kp first, then Kd)
PID_KP = 0.12   # proportional  (error in pixels → speed fraction)
PID_KD = 0.04   # derivative    (damps oscillation)
PID_KI = 0.0    # integral      (leave 0 unless steady-state drift observed)

# Deadzones
TURN_DEADZONE  = 40          # px — tighter than before for smoother tracking
DIST_DEADZONE_CM = 15        # cm — stop command issued within this band

# ──────────────────────────────────────────────────────────────────────────────
#  REAL DISTANCE: SHOULDER-WIDTH PINHOLE MODEL
#  Distance (cm) = (SHOULDER_WIDTH_CM × FOCAL_LEN_PX) / shoulder_px_width
#  Calibrate FOCAL_LEN_PX by measuring at a known distance once.
# ──────────────────────────────────────────────────────────────────────────────
SHOULDER_WIDTH_CM = 42.0     # average adult shoulder width
FOCAL_LEN_PX      = 600.0    # calibrate: at 1 m, typical shoulder spans ~250px
                               # → F = 250 × 100 / 42 ≈ 595  → set 600 as default
TARGET_DIST_CM    = 80.0      # desired following distance (cm)

# ──────────────────────────────────────────────────────────────────────────────
#  EMA SMOOTHING  (α near 1 = less smooth; near 0 = very smooth)
# ──────────────────────────────────────────────────────────────────────────────
EMA_ALPHA = 0.35    # for both cx and distance

# ──────────────────────────────────────────────────────────────────────────────
#  YOLO MODELS & KEYPOINT INDICES (COCO)
# ──────────────────────────────────────────────────────────────────────────────
VIDEO_URL = f"http://{PHONE_IP}:8080/video"

print("Loading YOLOv8n detection model...")
det_model  = YOLO('yolov8n.pt')

print("Loading YOLOv8n-pose model...")
pose_model = YOLO('yolov8n-pose.pt')

# COCO 17-keypoint indices
NOSE        = 0
SHOULDER_L  = 5
SHOULDER_R  = 6
WRIST_L     = 9
WRIST_R     = 10

# ──────────────────────────────────────────────────────────────────────────────
#  SHARED STATE
# ──────────────────────────────────────────────────────────────────────────────
latest_frame  = None
output_frame  = None
frame_lock    = threading.Lock()
stream_lock   = threading.Lock()

# Ring buffers for matplotlib telemetry (max N samples)
TELEM_LEN = 120
telem_lock = threading.Lock()
t_time     = collections.deque(maxlen=TELEM_LEN)
t_dist     = collections.deque(maxlen=TELEM_LEN)
t_offset   = collections.deque(maxlen=TELEM_LEN)
t_action   = collections.deque(maxlen=TELEM_LEN)   # str label
_telem_t0  = time.monotonic()


def _push_telem(dist_cm: float, offset_px: int, action: str):
    with telem_lock:
        t_time.append(time.monotonic() - _telem_t0)
        t_dist.append(dist_cm)
        t_offset.append(offset_px)
        t_action.append(action)


# ──────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB LIVE TELEMETRY (runs in main thread or a dedicated thread)
# ──────────────────────────────────────────────────────────────────────────────
def run_telemetry_plot():
    """
    Non-blocking matplotlib window showing:
      • Distance (cm) vs time
      • Horizontal offset (px) vs time
    Call this from a dedicated daemon thread.
    """
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), facecolor="#0d1117")
    fig.suptitle("Navis Tracker — Live Telemetry", color="#c9d1d9", fontsize=13, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    ax1.set_ylabel("Distance (cm)", color="#58a6ff")
    ax1.axhline(TARGET_DIST_CM, color="#f0883e", linewidth=1.2, linestyle="--", label=f"Target {TARGET_DIST_CM} cm")
    ax1.axhspan(TARGET_DIST_CM - DIST_DEADZONE_CM, TARGET_DIST_CM + DIST_DEADZONE_CM,
                alpha=0.12, color="#3fb950", label="Deadzone")
    ax1.legend(fontsize=8, labelcolor="#c9d1d9", facecolor="#161b22", edgecolor="#30363d")

    ax2.set_ylabel("Horiz. Offset (px)", color="#d2a8ff")
    ax2.axhline(0, color="#f0883e", linewidth=1.0, linestyle="--")
    ax2.axhspan(-TURN_DEADZONE, TURN_DEADZONE, alpha=0.12, color="#3fb950")
    ax2.set_xlabel("Time (s)", color="#8b949e")

    line1, = ax1.plot([], [], color="#58a6ff",  linewidth=1.5)
    line2, = ax2.plot([], [], color="#d2a8ff", linewidth=1.5)

    plt.tight_layout()
    plt.pause(0.01)

    while True:
        try:
            with telem_lock:
                xs     = list(t_time)
                ys_d   = list(t_dist)
                ys_off = list(t_offset)

            if len(xs) >= 2:
                line1.set_data(xs, ys_d)
                line2.set_data(xs, ys_off)
                for ax in (ax1, ax2):
                    ax.relim()
                    ax.autoscale_view()
                fig.canvas.flush_events()
                fig.canvas.draw_idle()

            time.sleep(0.15)
        except Exception:
            break


# ──────────────────────────────────────────────────────────────────────────────
#  ZERO-LAG FRAME GRABBER
#  Reads as fast as possible, always discarding stale frames.
#  The key trick: we do NOT yield (sleep) between reads — we drain the
#  MJPEG/RTSP buffer continuously so latest_frame is always fresh.
# ──────────────────────────────────────────────────────────────────────────────
def frame_grabber(url: str):
    global latest_frame
    print(f"Frame grabber connecting to {url}...")

    while True:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimize internal buffer

        if not cap.isOpened():
            print("Frame grabber: could not open stream. Retrying in 3 s...")
            time.sleep(3)
            continue

        print("Frame grabber: stream opened.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grabber: stream lost. Reconnecting in 2 s...")
                time.sleep(2)
                break

            with frame_lock:
                latest_frame = frame   # Swap pointer — O(1), no copy

        cap.release()


# ──────────────────────────────────────────────────────────────────────────────
#  MOVEMENT FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def _send_async(left: int, right: int):
    url = f"http://{PICO_IP}/move?left={left}&right={right}"
    try:
        requests.get(url, timeout=0.25)
    except requests.exceptions.RequestException:
        pass

def _send_stop_async():
    try:
        requests.get(f"http://{PICO_IP}/stop", timeout=0.5)
    except requests.exceptions.RequestException:
        pass

def send_speeds(left: int, right: int):
    threading.Thread(target=_send_async, args=(left, right), daemon=True).start()

def move_forward():  send_speeds( FORWARD_SPEED,   FORWARD_SPEED)
def move_backward(): send_speeds(-BACKWARD_SPEED, -BACKWARD_SPEED)
def stop():
    send_speeds(0, 0)
    threading.Thread(target=_send_stop_async, daemon=True).start()

def turn_pid(error_px: float, prev_error_px: float, dt: float) -> tuple[int, int]:
    """
    PID-based differential turn.  Returns (left_speed, right_speed).
    Positive error → person is to the RIGHT → robot turns RIGHT (left > right).
    """
    p = PID_KP * error_px
    d = PID_KD * (error_px - prev_error_px) / max(dt, 1e-3)
    raw = p + d  # negative for left, positive for right

    speed = int(min(abs(raw) * MAX_TURN_SPEED, MAX_TURN_SPEED))
    speed = max(speed, MIN_TURN_SPEED)         # min to overcome friction

    if raw > 0:          # turn right
        return  speed, -speed
    else:                # turn left
        return -speed,  speed


# ──────────────────────────────────────────────────────────────────────────────
#  HAND RAISE GESTURE
# ──────────────────────────────────────────────────────────────────────────────
def is_hand_raised(kpts_data) -> bool:
    """
    Returns True if ANY detected person has at least one wrist
    above their same-side shoulder (with confidence gates).
    kpts_data: tensor of shape [N_persons, 17, 3].
    """
    for kpts in kpts_data:
        lw_y, lw_c = float(kpts[WRIST_L][1]), float(kpts[WRIST_L][2])
        ls_y, ls_c = float(kpts[SHOULDER_L][1]), float(kpts[SHOULDER_L][2])
        rw_y, rw_c = float(kpts[WRIST_R][1]), float(kpts[WRIST_R][2])
        rs_y, rs_c = float(kpts[SHOULDER_R][1]), float(kpts[SHOULDER_R][2])

        left_raised  = lw_c > 0.4 and ls_c > 0.4 and lw_y < ls_y
        right_raised = rw_c > 0.4 and rs_c > 0.4 and rw_y < rs_y
        if left_raised or right_raised:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
#  REAL DISTANCE FROM SHOULDER WIDTH (pinhole camera model)
# ──────────────────────────────────────────────────────────────────────────────
def shoulder_distance_cm(kpts) -> float | None:
    """
    Compute distance from camera to person using shoulder keypoints.
    Returns None if keypoints are not reliable.
    """
    ls_x, ls_c = float(kpts[SHOULDER_L][0]), float(kpts[SHOULDER_L][2])
    rs_x, rs_c = float(kpts[SHOULDER_R][0]), float(kpts[SHOULDER_R][2])

    if ls_c < 0.4 or rs_c < 0.4:
        return None

    span_px = abs(rs_x - ls_x)
    if span_px < 10:        # too small / degenerate
        return None

    dist_cm = (SHOULDER_WIDTH_CM * FOCAL_LEN_PX) / span_px
    return dist_cm


def shoulder_midpoint_px(kpts) -> float | None:
    """Returns the x-pixel of the midpoint between both shoulders, or None."""
    ls_x, ls_c = float(kpts[SHOULDER_L][0]), float(kpts[SHOULDER_L][2])
    rs_x, rs_c = float(kpts[SHOULDER_R][0]), float(kpts[SHOULDER_R][2])
    if ls_c < 0.3 or rs_c < 0.3:
        return None
    return (ls_x + rs_x) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
#  SELECT BEST PERSON (largest bbox area = closest, most reliable)
# ──────────────────────────────────────────────────────────────────────────────
def select_best_box(boxes):
    """
    Given YOLO boxes tensor, return the index of the box with the
    largest area (proxy for closest, most-prominent person).
    """
    best_idx  = 0
    best_area = -1
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_idx  = i
    return best_idx


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN VISION + CONTROL LOOP
# ──────────────────────────────────────────────────────────────────────────────
def vision_loop():
    global output_frame

    # EMA state
    ema_cx   = None
    ema_dist = None

    # PID state
    prev_error = 0.0
    prev_time  = time.monotonic()

    last_left  = 0
    last_right = 0
    frames_since_cmd = 0
    pose_every = 3          # run pose every N frames (saves ~40 ms/frame)
    frame_counter = 0

    print("Vision loop started — waiting for first frame...")

    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()

        frame_counter += 1
        now = time.monotonic()
        dt  = now - prev_time
        prev_time = now

        h, w, _ = frame.shape
        frame_cx  = w // 2
        annotated = frame.copy()

        action_label = "SEARCHING..."
        label_color  = (200, 200, 200)
        next_left = next_right = 0

        # ── Run pose model (every N frames) ─────────────────────────────────
        pose_results = None
        if frame_counter % pose_every == 0:
            pose_results = pose_model(frame, verbose=False)

        # ── Hand raise check ─────────────────────────────────────────────────
        if pose_results is not None and len(pose_results[0].keypoints) > 0:
            if is_hand_raised(pose_results[0].keypoints.data):
                stop()
                action_label = "STOP — Hand Raised"
                label_color  = (0, 0, 255)
                cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 180), -1)
                cv2.putText(annotated, "✋  " + action_label,
                            (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                with stream_lock:
                    output_frame = annotated.copy()
                _push_telem(ema_dist or 0, 0, action_label)
                continue

        # ── Detection ────────────────────────────────────────────────────────
        det_results = det_model(frame, classes=[0], verbose=False)

        if len(det_results[0].boxes) == 0:
            stop()
            ema_cx = ema_dist = None          # reset smoothing on lost target
            prev_error = 0.0
            action_label = "NO HUMAN — STOPPED"
            label_color  = (0, 120, 255)
            cv2.putText(annotated, action_label,
                        (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, label_color, 2)
            with stream_lock:
                output_frame = annotated.copy()
            _push_telem(0, 0, action_label)
            continue

        # ── Pick best person ─────────────────────────────────────────────────
        best_idx = select_best_box(det_results[0].boxes)
        box = det_results[0].boxes[best_idx].xyxy[0]
        x1, y1, x2, y2 = map(int, box)

        # Draw all boxes dimly, best person brightly
        for i, b in enumerate(det_results[0].boxes):
            bx1, by1, bx2, by2 = map(int, b.xyxy[0])
            color = (0, 255, 80) if i == best_idx else (80, 80, 80)
            thickness = 2 if i == best_idx else 1
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, thickness)

        # ── Shoulder midpoint as tracking reference (via pose) ───────────────
        raw_cx = (x1 + x2) // 2   # default: bbox centre

        dist_cm = None
        if pose_results is not None and len(pose_results[0].keypoints) > best_idx:
            kpts = pose_results[0].keypoints.data[best_idx]
            mid  = shoulder_midpoint_px(kpts)
            if mid is not None:
                raw_cx = int(mid)
                # Draw shoulder midpoint marker
                mid_y  = int(float(kpts[SHOULDER_L][1]) + float(kpts[SHOULDER_R][1])) // 2
                cv2.circle(annotated, (raw_cx, mid_y), 7, (255, 200, 0), -1)
                cv2.putText(annotated, "SHL MID", (raw_cx - 30, mid_y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

            d = shoulder_distance_cm(kpts)
            if d is not None:
                dist_cm = d

        # ── EMA smoothing ────────────────────────────────────────────────────
        if ema_cx is None:
            ema_cx = float(raw_cx)
        else:
            ema_cx = EMA_ALPHA * raw_cx + (1 - EMA_ALPHA) * ema_cx

        # Fallback distance via bbox height ratio
        if dist_cm is None:
            box_h = y2 - y1
            ratio = box_h / float(h)
            dist_cm = (0.60 / ratio) * TARGET_DIST_CM  # rough estimate

        if ema_dist is None:
            ema_dist = dist_cm
        else:
            ema_dist = EMA_ALPHA * dist_cm + (1 - EMA_ALPHA) * ema_dist

        smooth_cx  = int(ema_cx)
        smooth_dist = ema_dist
        horiz_offset = smooth_cx - frame_cx     # +ve = person is RIGHT of centre
        dist_error   = smooth_dist - TARGET_DIST_CM   # +ve = too far

        # ── Draw tracking overlay ────────────────────────────────────────────
        cv2.line(annotated, (frame_cx, 0), (frame_cx, h), (60, 60, 60), 1)  # centre guideline
        cv2.line(annotated, (frame_cx, h // 2), (smooth_cx, h // 2), (0, 255, 80), 2)
        cv2.circle(annotated, (smooth_cx, (y1 + y2) // 2), 6, (0, 255, 80), -1)

        # Distance bar on the right edge
        bar_max_h = h - 80
        bar_fill  = int(min(smooth_dist / (TARGET_DIST_CM * 2), 1.0) * bar_max_h)
        cv2.rectangle(annotated, (w - 25, 40), (w - 10, 40 + bar_max_h), (40, 40, 40), -1)
        bar_color = (0, 255, 80) if abs(dist_error) < DIST_DEADZONE_CM else (0, 120, 255)
        cv2.rectangle(annotated, (w - 25, 40 + bar_max_h - bar_fill),
                      (w - 10, 40 + bar_max_h), bar_color, -1)
        cv2.putText(annotated, f"{smooth_dist:.0f}cm",
                    (w - 55, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

        # ── PID turning + distance control ──────────────────────────────────
        if abs(horiz_offset) > TURN_DEADZONE:
            next_left, next_right = turn_pid(horiz_offset, prev_error, dt)
            prev_error = horiz_offset
            direction  = "RIGHT" if horiz_offset > 0 else "LEFT"
            action_label = f"TURNING {direction} ({horiz_offset:+d}px)"
            label_color  = (0, 200, 255) if horiz_offset > 0 else (255, 200, 0)
            send_speeds(next_left, next_right)

        elif abs(dist_error) > DIST_DEADZONE_CM:
            prev_error = 0.0    # reset turn integral while driving straight
            if dist_error > 0:  # too far → move forward
                move_forward()
                next_left = next_right = FORWARD_SPEED
                action_label = f"FORWARD  {smooth_dist:.0f} cm  (target {TARGET_DIST_CM:.0f})"
                label_color  = (0, 255, 80)
            else:               # too close → move backward
                move_backward()
                next_left = next_right = -BACKWARD_SPEED
                action_label = f"BACKWARD {smooth_dist:.0f} cm  (target {TARGET_DIST_CM:.0f})"
                label_color  = (0, 80, 255)
        else:
            prev_error = 0.0
            stop()
            action_label = f"HOLDING  {smooth_dist:.0f} cm  ✓"
            label_color  = (255, 255, 255)

        # ── HUD ──────────────────────────────────────────────────────────────
        cv2.rectangle(annotated, (0, 0), (w, 58), (15, 15, 15), -1)
        cv2.putText(annotated, action_label,
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
        cv2.putText(annotated,
                    f"L:{next_left:+d}  R:{next_right:+d} | dist:{smooth_dist:.1f}cm "
                    f"| x_off:{horiz_offset:+d}px | ema_cx:{smooth_cx}",
                    (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

        _push_telem(smooth_dist, horiz_offset, action_label)

        # ── Keepalive (resend to avoid motor coast on dropped packets) ───────
        changed = (abs(next_left - last_left) > 5 or abs(next_right - last_right) > 5)
        if changed or frames_since_cmd > 6:
            send_speeds(next_left, next_right)
            last_left, last_right = next_left, next_right
            frames_since_cmd = 0
        else:
            frames_since_cmd += 1

        with stream_lock:
            output_frame = annotated.copy()


# ──────────────────────────────────────────────────────────────────────────────
#  MJPEG STREAMING SERVER  (port 5001)
# ──────────────────────────────────────────────────────────────────────────────
server_app = Flask(__name__)

def generate_feed():
    while True:
        with stream_lock:
            frame = output_frame

        if frame is None:
            time.sleep(0.04)
            continue

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            time.sleep(0.04)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)   # ~30 fps to client


@server_app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Navis Tracker</title>
        <meta charset="utf-8">
        <style>
          body{background:#0d1117;color:#c9d1d9;text-align:center;
               font-family:'Segoe UI',sans-serif;padding:30px}
          img{max-width:95%;border:2px solid #4facfe;border-radius:12px;
              box-shadow:0 0 24px #4facfe44}
          h2{color:#4facfe;letter-spacing:2px}
        </style>
      </head>
      <body>
        <h2>🤖 Navis Robot — Vision Feed</h2>
        <img src="/video_feed">
      </body>
    </html>
    """

@server_app.route("/video_feed")
def video_feed():
    return Response(generate_feed(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def _emergency_stop():
    print("[SHUTDOWN] Sending stop to Pico before exit...")
    _send_stop_async()


if __name__ == '__main__':
    atexit.register(_emergency_stop)
    signal.signal(signal.SIGTERM, lambda *_: exit(0))

    # Zero-lag frame grabber thread
    threading.Thread(target=frame_grabber, args=(VIDEO_URL,), daemon=True).start()

    # Give the grabber a moment to connect before inference starts
    time.sleep(2)

    # Vision+control loop
    threading.Thread(target=vision_loop, daemon=True).start()

    # Matplotlib telemetry (daemon so it dies with the process)
    threading.Thread(target=run_telemetry_plot, daemon=True).start()

    print("Navis tracker started — streaming on http://0.0.0.0:5001")
    server_app.run(host="0.0.0.0", port=5001, debug=False,
                   threaded=True, use_reloader=False)
