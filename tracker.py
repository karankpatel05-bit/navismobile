import atexit
import cv2
import os
import requests
import signal
import threading
import time
from flask import Flask, Response
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv('env')

PHONE_IP = os.getenv("PHONE_IP", "")
PICO_IP  = os.getenv("PICO_IP", "")

if not PHONE_IP or not PICO_IP:
    print("Error: PHONE_IP and PICO_IP must be set in the 'env' file.")
    exit(1)

# ─────────────────────────────────────────────────────────
#  SPEED CONSTANTS  — kept deliberately low for safety
#  Raise these only after confirming basic tracking works.
# ─────────────────────────────────────────────────────────
FORWARD_SPEED  = 100   # Both wheels forward (was 120)
BACKWARD_SPEED = 90    # Both wheels backward (was 110)
TURN_SPEED     = 85    # Differential turn (was 100)

# Deadzones — how much slack before the robot reacts
TURN_DEADZONE  = 100    # pixels from center before turning
DIST_DEADZONE  = 0.08  # fraction of frame height error before moving

# Target: person fills ~60% of frame height ≈ 10 cm comfort distance
TARGET_HEIGHT_RATIO = 0.60

# ─────────────────────────────────────────────────────────
#  YOLO MODELS
# ─────────────────────────────────────────────────────────
VIDEO_URL = f"http://{PHONE_IP}:8080/video"

print("Loading YOLOv8n detection model...")
det_model  = YOLO('yolov8n.pt')

print("Loading YOLOv8n-pose model...")
pose_model = YOLO('yolov8n-pose.pt')

# COCO pose keypoint indices
SHOULDER_L, SHOULDER_R = 5, 6
WRIST_L,    WRIST_R    = 9, 10

# ─────────────────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────────────────
# latest_frame is always the freshest raw frame from the camera.
# output_frame is the annotated frame for the MJPEG stream.
latest_frame  = None
output_frame  = None
frame_lock    = threading.Lock()
stream_lock   = threading.Lock()


# ─────────────────────────────────────────────────────────
#  CRITICAL FIX: DEDICATED FRAME GRABBER THREAD
#  This thread runs as fast as possible, constantly calling
#  cap.read() and DISCARDING old frames.  This keeps
#  latest_frame always fresh (< 50 ms old) regardless of
#  how slow YOLO inference is, eliminating the 5-second lag.
# ─────────────────────────────────────────────────────────
def frame_grabber(url: str):
    global latest_frame
    print(f"Frame grabber connecting to {url}...")

    while True:
        cap = cv2.VideoCapture(url)
        # Tell OpenCV to keep the buffer as small as possible
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("Frame grabber: could not open stream. Retrying in 3s...")
            time.sleep(3)
            continue

        print("Frame grabber: stream opened.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grabber: stream lost. Reconnecting in 2s...")
                time.sleep(2)
                break   # Break inner loop → reconnect outer loop

            # Only hold the lock long enough to swap the pointer
            with frame_lock:
                latest_frame = frame  # Always the most recent frame

        cap.release()


# ─────────────────────────────────────────────────────────
#  MOVEMENT FUNCTIONS
# ─────────────────────────────────────────────────────────
def _send_async(left: int, right: int):
    url = f"http://{PICO_IP}/move?left={left}&right={right}"
    try:
        requests.get(url, timeout=0.25)
    except requests.exceptions.RequestException:
        pass

def _send_stop_async():
    """Hits the Pico /stop endpoint directly. Used for safety since the
    firmware no longer has its own watchdog timeout."""
    try:
        requests.get(f"http://{PICO_IP}/stop", timeout=0.5)
    except requests.exceptions.RequestException:
        pass

def send_speeds(left: int, right: int):
    threading.Thread(target=_send_async, args=(left, right), daemon=True).start()

def move_forward():  send_speeds( FORWARD_SPEED,   FORWARD_SPEED)
def move_backward(): send_speeds(-BACKWARD_SPEED, -BACKWARD_SPEED)
def turn_left():     send_speeds(-TURN_SPEED,       TURN_SPEED)
def turn_right():    send_speeds( TURN_SPEED,      -TURN_SPEED)
def stop():
    """Send zero speeds AND hit /stop — critical since firmware has no watchdog."""
    send_speeds(0, 0)
    threading.Thread(target=_send_stop_async, daemon=True).start()


# ─────────────────────────────────────────────────────────
#  HAND RAISE GESTURE
# ─────────────────────────────────────────────────────────
def is_hand_raised(pose_results) -> bool:
    if not pose_results or len(pose_results[0].keypoints) == 0:
        return False
    for kpts in pose_results[0].keypoints.data:
        lw_y, lw_c = float(kpts[WRIST_L][1]),   float(kpts[WRIST_L][2])
        ls_y, ls_c = float(kpts[SHOULDER_L][1]), float(kpts[SHOULDER_L][2])
        rw_y, rw_c = float(kpts[WRIST_R][1]),   float(kpts[WRIST_R][2])
        rs_y, rs_c = float(kpts[SHOULDER_R][1]), float(kpts[SHOULDER_R][2])

        left_raised  = lw_c > 0.4 and ls_c > 0.4 and lw_y < ls_y
        right_raised = rw_c > 0.4 and rs_c > 0.4 and rw_y < rs_y
        if left_raised or right_raised:
            return True
    return False


# ─────────────────────────────────────────────────────────
#  MAIN VISION + CONTROL LOOP
# ─────────────────────────────────────────────────────────
def vision_loop():
    global output_frame

    last_left  = 0
    last_right = 0
    frames_since_cmd  = 0
    pose_check_every  = 5   # Run pose model every N frames (saves CPU)
    frame_counter     = 0

    print("Vision loop started. Waiting for first frame...")

    while True:
        # ── Get the LATEST frame (never a stale buffered one) ─────────
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()  # Safe copy; grabber can keep running

        frame_counter += 1
        h, w, _ = frame.shape
        frame_cx = w // 2
        annotated = frame.copy()
        action_label = "SEARCHING..."
        label_color  = (200, 200, 200)
        next_left = next_right = 0

        # ── Pose check (every N frames to save CPU) ────────────────────
        hand_raised = False
        if frame_counter % pose_check_every == 0:
            pose_results = pose_model(frame, verbose=False)
            hand_raised  = is_hand_raised(pose_results)

        if hand_raised:
            stop()
            action_label = "STOP — Hand Raised"
            label_color  = (0, 0, 255)
            cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 180), -1)
            cv2.putText(annotated, "✋  " + action_label,
                        (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            with stream_lock:
                output_frame = annotated.copy()
            continue

        # ── Detection ─────────────────────────────────────────────────
        det_results = det_model(frame, classes=[0], verbose=False)

        if len(det_results[0].boxes) == 0:
            stop()
            action_label = "NO HUMAN — STOPPED"
            label_color  = (0, 120, 255)
            cv2.putText(annotated, action_label,
                        (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, label_color, 2)
            with stream_lock:
                output_frame = annotated.copy()
            continue

        # ── Bounding box analysis ──────────────────────────────────────
        box = det_results[0].boxes[0].xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 80), 2)

        box_h        = y2 - y1
        person_cx    = (x1 + x2) // 2
        horiz_offset = person_cx - frame_cx     # + = person is RIGHT of centre
        cur_ratio    = box_h / float(h)
        dist_error   = TARGET_HEIGHT_RATIO - cur_ratio  # + = too far

        approx_cm = max(1, int((TARGET_HEIGHT_RATIO / cur_ratio) * 10))

        cv2.line(annotated, (frame_cx, h // 2), (person_cx, h // 2), (0, 255, 80), 2)
        cv2.circle(annotated, (person_cx, (y1 + y2) // 2), 6, (0, 255, 80), -1)

        # ── Direction decision ────────────────────────────────────────
        #  Priority: YAW correction first, then DISTANCE correction.
        if abs(horiz_offset) > TURN_DEADZONE:
            if horiz_offset > 0:
                turn_right()
                next_left, next_right = TURN_SPEED, -TURN_SPEED
                action_label = f"TURNING RIGHT ({horiz_offset:+d}px)"
                label_color  = (0, 200, 255)
            else:
                turn_left()
                next_left, next_right = -TURN_SPEED, TURN_SPEED
                action_label = f"TURNING LEFT ({horiz_offset:+d}px)"
                label_color  = (255, 200, 0)

        elif abs(dist_error) > DIST_DEADZONE:
            if dist_error > 0:
                move_forward()
                next_left = next_right = FORWARD_SPEED
                action_label = f"FORWARD (~{approx_cm}cm)"
                label_color  = (0, 255, 80)
            else:
                move_backward()
                next_left = next_right = -BACKWARD_SPEED
                action_label = f"BACKWARD (~{approx_cm}cm)"
                label_color  = (0, 80, 255)
        else:
            stop()
            action_label = f"HOLDING (~{approx_cm}cm)"
            label_color  = (255, 255, 255)

        # ── HUD overlay ───────────────────────────────────────────────
        cv2.rectangle(annotated, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(annotated, action_label,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
        cv2.putText(annotated,
                    f"L:{next_left:+d}  R:{next_right:+d} | "
                    f"ratio:{cur_ratio:.2f} | x_off:{horiz_offset:+d}px | ~{approx_cm}cm",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # ── Periodic keepalive ────────────────────────────────────────
        # NOTE: Firmware watchdog removed — we resend periodically anyway
        # so the motors don't coast on a lost/dropped UDP-style HTTP packet.
        changed = (abs(next_left - last_left) > 5 or abs(next_right - last_right) > 5)
        if changed or frames_since_cmd > 8:
            send_speeds(next_left, next_right)
            last_left, last_right = next_left, next_right
            frames_since_cmd = 0
        else:
            frames_since_cmd += 1

        with stream_lock:
            output_frame = annotated.copy()


# ─────────────────────────────────────────────────────────
#  MJPEG STREAMING SERVER  (port 5001)
# ─────────────────────────────────────────────────────────
server_app = Flask(__name__)

def generate_feed():
    global output_frame
    while True:
        with stream_lock:
            frame = output_frame

        if frame is None:
            time.sleep(0.04)
            continue  # Correctly outside the lock now

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            time.sleep(0.04)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.04)  # ~25 fps cap


@server_app.route("/")
def index():
    return """
    <html>
      <head><title>Navis Tracker</title></head>
      <body style="background:#0b0c10;color:#4facfe;text-align:center;font-family:sans-serif;padding-top:40px">
        <h2>Navis Robot Vision Feed</h2>
        <img src="/video_feed" style="max-width:95%;border:2px solid #4facfe;border-radius:10px">
      </body>
    </html>
    """

@server_app.route("/video_feed")
def video_feed():
    return Response(generate_feed(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def _emergency_stop():
    """Called on clean exit or Ctrl+C — stops motors since firmware has no watchdog."""
    print("[SHUTDOWN] Sending stop to Pico before exit...")
    _send_stop_async()

if __name__ == '__main__':
    atexit.register(_emergency_stop)
    signal.signal(signal.SIGTERM, lambda *_: exit(0))  # triggers atexit on kill

    # Start the dedicated frame grabber (kills latency)
    threading.Thread(target=frame_grabber, args=(VIDEO_URL,), daemon=True).start()

    # Give the grabber a moment to connect before inference starts
    time.sleep(2)

    # Start the YOLO tracking loop
    threading.Thread(target=vision_loop, daemon=True).start()

    print("Navis tracker started — streaming on http://0.0.0.0:5001")
    server_app.run(host="0.0.0.0", port=5001, debug=False,
                   threaded=True, use_reloader=False)

