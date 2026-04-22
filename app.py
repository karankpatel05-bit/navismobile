import atexit
import cv2
import os
import requests
import signal
import threading
import time
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
from ultralytics import YOLO
from groq import Groq

# Re-use existing database logic for local training override
from database import load_training_data, add_qa_pair, delete_qa_pair, init_storage
from difflib import SequenceMatcher

# Load environment variables
load_dotenv('env')

PHONE_IP = os.getenv("PHONE_IP", "")
PICO_IP = os.getenv("PICO_IP", "")
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

app = Flask(__name__)

# ─────────────────────────────────────────────────────────
#  CHAT & AI CONSTANTS
# ─────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.72
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are Navis, a friendly, AI robotics assistant running on a mobile platform.
Your personality should be professional, cheerful, and approachable.
You are the face and brain of a Differential Drive robot.
IMPORTANT: You understand spoken context. Keep all your responses concise (2-3 sentences max) because your text will always be read aloud via Text-to-Speech by the robot. Be conversational and skip Markdown formatting entirely.
"""

client = None
conversation_history = []
active_mode = "chatbot" # "chatbot" or "follower"

def init_groq():
    global client
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)

def chat_with_groq(message):
    global conversation_history
    conversation_history.append({"role": "user", "content": message})
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
    )
    
    assistant_text = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_text})
    
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
        
    return assistant_text

def find_matching_qa(question):
    data = load_training_data()
    q_lower = question.lower().strip()
    best_match = None
    best_score = 0

    for qa in data.get('qa_pairs', []):
        trained_q = qa['question'].lower().strip()
        seq_score = SequenceMatcher(None, q_lower, trained_q).ratio()
        t_words = set(trained_q.split())
        q_words = set(q_lower.split())
        overlap = len(t_words & q_words) / max(len(t_words), 1)
        
        combined = (seq_score + overlap) / 2
        if combined > best_score:
            best_score = combined
            best_match = qa

    if best_score >= SIMILARITY_THRESHOLD and best_match:
        return best_match['answer']
    return None

# ─────────────────────────────────────────────────────────
#  ROBOT VISION & MOVEMENT CONSTANTS
# ─────────────────────────────────────────────────────────
FORWARD_SPEED  = 80
BACKWARD_SPEED = 80
MAX_TURN_SPEED = 85
MIN_TURN_SPEED = 40

# PID gains for yaw (tune Kp first)
PID_KP = 0.12
PID_KD = 0.04

TURN_DEADZONE    = 40       # px
DIST_DEADZONE_CM = 15       # cm

# Real distance model — pinhole camera using shoulder width
SHOULDER_WIDTH_CM = 42.0
FOCAL_LEN_PX      = 600.0
TARGET_DIST_CM    = 80.0

# EMA smoothing coefficient (0 < α ≤ 1)
EMA_ALPHA = 0.35

VIDEO_URL = f"http://{PHONE_IP}:8080/video"

if PHONE_IP and PICO_IP:
    print("Loading YOLOv8n models...")
    det_model  = YOLO('yolov8n.pt')
    pose_model = YOLO('yolov8n-pose.pt')
else:
    print("WARNING: Missing PHONE_IP or PICO_IP. Tracking will not function.")
    det_model = None
    pose_model = None

# COCO 17-keypoint indices
NOSE       = 0
SHOULDER_L = 5
SHOULDER_R = 6
WRIST_L    = 9
WRIST_R    = 10

latest_frame  = None
output_frame  = None
frame_lock    = threading.Lock()
stream_lock   = threading.Lock()

# ─────────────────────────────────────────────────────────
#  FRAME GRABBER & VISION LOOP
# ─────────────────────────────────────────────────────────
def frame_grabber(url: str):
    global latest_frame
    print(f"Frame grabber connecting to {url}...")
    while True:
        cap = cv2.VideoCapture(url)
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
                break
            with frame_lock:
                latest_frame = frame 
        cap.release()

def _send_async(left: int, right: int):
    if not PICO_IP: return
    url = f"http://{PICO_IP}/move?left={left}&right={right}"
    try:
        requests.get(url, timeout=0.25)
    except requests.exceptions.RequestException:
        pass

def _send_stop_async():
    if not PICO_IP: return
    try:
        requests.get(f"http://{PICO_IP}/stop", timeout=0.5)
        print("[ESTOP/SAFETY] Stop command sent to Pico")
    except Exception:
        pass

def send_speeds(left: int, right: int):
    threading.Thread(target=_send_async, args=(left, right), daemon=True).start()

def move_forward():  send_speeds( FORWARD_SPEED,   FORWARD_SPEED)
def move_backward(): send_speeds(-BACKWARD_SPEED, -BACKWARD_SPEED)
def stop():
    send_speeds(0, 0)
    threading.Thread(target=_send_stop_async, daemon=True).start()

def turn_pid(error_px: float, prev_error: float, dt: float):
    """PID turn: returns (left_speed, right_speed)."""
    p = PID_KP * error_px
    d = PID_KD * (error_px - prev_error) / max(dt, 1e-3)
    raw = p + d
    speed = int(min(abs(raw) * MAX_TURN_SPEED, MAX_TURN_SPEED))
    speed = max(speed, MIN_TURN_SPEED)
    return (speed, -speed) if raw > 0 else (-speed, speed)

def is_hand_raised(kpts_data) -> bool:
    for kpts in kpts_data:
        lw_y, lw_c = float(kpts[WRIST_L][1]), float(kpts[WRIST_L][2])
        ls_y, ls_c = float(kpts[SHOULDER_L][1]), float(kpts[SHOULDER_L][2])
        rw_y, rw_c = float(kpts[WRIST_R][1]), float(kpts[WRIST_R][2])
        rs_y, rs_c = float(kpts[SHOULDER_R][1]), float(kpts[SHOULDER_R][2])
        if (lw_c > 0.4 and ls_c > 0.4 and lw_y < ls_y) or \
           (rw_c > 0.4 and rs_c > 0.4 and rw_y < rs_y):
            return True
    return False

def shoulder_distance_cm(kpts):
    ls_x, ls_c = float(kpts[SHOULDER_L][0]), float(kpts[SHOULDER_L][2])
    rs_x, rs_c = float(kpts[SHOULDER_R][0]), float(kpts[SHOULDER_R][2])
    if ls_c < 0.4 or rs_c < 0.4:
        return None
    span = abs(rs_x - ls_x)
    return (SHOULDER_WIDTH_CM * FOCAL_LEN_PX) / span if span > 10 else None

def shoulder_midpoint_px(kpts):
    ls_x, ls_c = float(kpts[SHOULDER_L][0]), float(kpts[SHOULDER_L][2])
    rs_x, rs_c = float(kpts[SHOULDER_R][0]), float(kpts[SHOULDER_R][2])
    if ls_c < 0.3 or rs_c < 0.3:
        return None
    return (ls_x + rs_x) / 2.0

def select_best_box(boxes):
    best_idx, best_area = 0, -1
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_idx = i
    return best_idx

def vision_loop():
    global output_frame, active_mode
    if not det_model:
        return

    # EMA state
    ema_cx = ema_dist = None
    # PID state
    prev_error = 0.0
    prev_time  = time.monotonic()

    last_left, last_right = 0, 0
    frames_since_cmd = 0
    pose_every   = 3
    frame_counter = 0

    print("Vision loop started. Waiting for first frame...")
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()

        if active_mode != "follower":
            time.sleep(0.1)
            continue

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

        # Pose check every N frames
        pose_results = None
        if frame_counter % pose_every == 0:
            pose_results = pose_model(frame, verbose=False)

        # Hand-raise check
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
                continue

        det_results = det_model(frame, classes=[0], verbose=False)

        if len(det_results[0].boxes) == 0:
            stop()
            ema_cx = ema_dist = None
            prev_error = 0.0
            action_label = "NO HUMAN — STOPPED"
            label_color  = (0, 120, 255)
            cv2.putText(annotated, action_label,
                        (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, label_color, 2)
            with stream_lock:
                output_frame = annotated.copy()
            continue

        # Pick closest / largest person
        best_idx = select_best_box(det_results[0].boxes)
        box = det_results[0].boxes[best_idx].xyxy[0]
        x1, y1, x2, y2 = map(int, box)

        for i, b in enumerate(det_results[0].boxes):
            bx1, by1, bx2, by2 = map(int, b.xyxy[0])
            color = (0, 255, 80) if i == best_idx else (80, 80, 80)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color,
                          2 if i == best_idx else 1)

        raw_cx  = (x1 + x2) // 2
        dist_cm = None

        if pose_results is not None and len(pose_results[0].keypoints) > best_idx:
            kpts = pose_results[0].keypoints.data[best_idx]
            mid  = shoulder_midpoint_px(kpts)
            if mid is not None:
                raw_cx = int(mid)
                mid_y  = int(float(kpts[SHOULDER_L][1]) +
                             float(kpts[SHOULDER_R][1])) // 2
                cv2.circle(annotated, (raw_cx, mid_y), 7, (255, 200, 0), -1)
            dist_cm = shoulder_distance_cm(kpts)

        # EMA smoothing
        ema_cx = (EMA_ALPHA * raw_cx + (1 - EMA_ALPHA) * ema_cx
                  if ema_cx is not None else float(raw_cx))

        if dist_cm is None:
            # Fallback: use bounding-box WIDTH as a proxy for shoulder span
            # Same pinhole model as keypoint path: D = (W_cm * F_px) / span_px
            box_w   = x2 - x1
            dist_cm = (SHOULDER_WIDTH_CM * FOCAL_LEN_PX) / box_w if box_w > 10 else TARGET_DIST_CM

        ema_dist = (EMA_ALPHA * dist_cm + (1 - EMA_ALPHA) * ema_dist
                    if ema_dist is not None else dist_cm)

        smooth_cx    = int(ema_cx)
        smooth_dist  = ema_dist
        horiz_offset = smooth_cx - frame_cx
        dist_error   = smooth_dist - TARGET_DIST_CM

        # Tracking overlay
        cv2.line(annotated, (frame_cx, 0), (frame_cx, h), (60, 60, 60), 1)
        cv2.line(annotated, (frame_cx, h // 2), (smooth_cx, h // 2), (0, 255, 80), 2)
        cv2.circle(annotated, (smooth_cx, (y1 + y2) // 2), 6, (0, 255, 80), -1)

        # Distance bar
        bar_max_h = h - 80
        bar_fill  = int(min(smooth_dist / (TARGET_DIST_CM * 2), 1.0) * bar_max_h)
        bar_color = (0, 255, 80) if abs(dist_error) < DIST_DEADZONE_CM else (0, 120, 255)
        cv2.rectangle(annotated, (w - 25, 40), (w - 10, 40 + bar_max_h), (40, 40, 40), -1)
        cv2.rectangle(annotated,
                      (w - 25, 40 + bar_max_h - bar_fill),
                      (w - 10, 40 + bar_max_h), bar_color, -1)
        cv2.putText(annotated, f"{smooth_dist:.0f}cm",
                    (w - 55, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

        # PID yaw + distance control
        if abs(horiz_offset) > TURN_DEADZONE:
            next_left, next_right = turn_pid(horiz_offset, prev_error, dt)
            prev_error = horiz_offset
            direction  = "RIGHT" if horiz_offset > 0 else "LEFT"
            action_label = f"TURNING {direction} ({horiz_offset:+d}px)"
            label_color  = (0, 200, 255) if horiz_offset > 0 else (255, 200, 0)
            send_speeds(next_left, next_right)
        elif abs(dist_error) > DIST_DEADZONE_CM:
            prev_error = 0.0
            if dist_error > 0:
                move_forward()
                next_left = next_right = FORWARD_SPEED
                action_label = f"FORWARD  {smooth_dist:.0f}cm (target {TARGET_DIST_CM:.0f})"
                label_color  = (0, 255, 80)
            else:
                move_backward()
                next_left = next_right = -BACKWARD_SPEED
                action_label = f"BACKWARD {smooth_dist:.0f}cm (target {TARGET_DIST_CM:.0f})"
                label_color  = (0, 80, 255)
        else:
            prev_error = 0.0
            stop()
            action_label = f"HOLDING  {smooth_dist:.0f}cm ✓"
            label_color  = (255, 255, 255)

        # HUD
        cv2.rectangle(annotated, (0, 0), (w, 58), (15, 15, 15), -1)
        cv2.putText(annotated, action_label,
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
        cv2.putText(annotated,
                    f"L:{next_left:+d}  R:{next_right:+d} | dist:{smooth_dist:.1f}cm "
                    f"| x_off:{horiz_offset:+d}px | ema_cx:{smooth_cx}",
                    (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

        changed = (abs(next_left - last_left) > 5 or abs(next_right - last_right) > 5)
        if changed or frames_since_cmd > 6:
            send_speeds(next_left, next_right)
            last_left, last_right = next_left, next_right
            frames_since_cmd = 0
        else:
            frames_since_cmd += 1

        with stream_lock:
            output_frame = annotated.copy()

def generate_feed():
    global output_frame
    while True:
        with stream_lock:
            frame = output_frame
        if frame is None:
            time.sleep(0.04)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            time.sleep(0.04)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.04)

# ─────────────────────────────────────────────────────────
#  WEB APIs & Flask Routes
# ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Empty message'}), 400

    trained_answer = find_matching_qa(message)
    if trained_answer:
        return jsonify({'response': trained_answer, 'source': 'trained'})
        
    if not client:
        return jsonify({'error': 'GROQ API key not configured', 'response': 'I am missing my brain connection! Check GROQ API Key in env.'})
        
    try:
        response_text = chat_with_groq(message)
        return jsonify({'response': response_text, 'source': 'ai'})
    except Exception as e:
        return jsonify({'response': f"Sorry, I encountered an error: {str(e)}", 'source': 'error'}), 500

@app.route('/api/mouth', methods=['POST'])
def mouth():
    global active_mode
    # Disallow mouth syncing if not exactly in chatbot mode
    if active_mode != "chatbot":
        return jsonify({'success': False, 'status': 'skipped_wrong_mode'})

    data = request.json
    raw_state = data.get('state', 'stop')
    pico_state = 'start' if raw_state in [1, '1', 'start'] else 'stop'
    
    if not PICO_IP:
        return jsonify({'success': False, 'error': 'No PICO_IP set'})
        
    def _send_proxy():
        try:
            requests.get(f"http://{PICO_IP}/mouth?state={pico_state}", timeout=0.5)
        except Exception:
            pass
            
    threading.Thread(target=_send_proxy, daemon=True).start()
    return jsonify({'success': True, 'state': pico_state})

@app.route('/api/mode', methods=['POST'])
def set_mode():
    global active_mode
    data = request.json
    new_mode = data.get('mode')
    
    if new_mode in ["chatbot", "follower", "manual"]:
        old_mode = active_mode
        active_mode = new_mode
        print(f"[MODE SWITCH] Changed from {old_mode} to {active_mode}")
        
        if old_mode == "follower" and active_mode != "follower":
            # Force stop wheels when leaving Follower/Manual mode
            print("[SAFETY] Force stopping wheels because exiting Follower mode...")
            stop()
        elif old_mode == "manual" and active_mode != "manual":
            print("[SAFETY] Force stopping wheels because exiting Manual mode...")
            stop()
            
        return jsonify({'success': True, 'mode': active_mode})
    return jsonify({'error': 'Invalid mode'}), 400

@app.route('/api/manual', methods=['POST'])
def manual_control():
    """D-pad manual control — maps action names to (left, right) motor speeds."""
    if active_mode != "manual":
        return jsonify({'success': False, 'status': 'skipped_wrong_mode'})

    data   = request.json or {}
    action = data.get('action', 'stop')
    speed  = int(data.get('speed', 80))
    speed  = max(0, min(speed, 150))  # clamp to safe range

    half = speed // 2  # slower wheel for turns

    ACTION_MAP = {
        'forward':   ( speed,  speed),
        'backward':  (-speed, -speed),
        'left':      (-half,   speed),
        'right':     ( speed, -half),
        'fwd_left':  ( half,   speed),
        'fwd_right': ( speed,  half),
        'bwd_left':  (-speed, -half),
        'bwd_right': (-half,  -speed),
        'stop':      (0,       0),
    }

    left, right = ACTION_MAP.get(action, (0, 0))
    send_speeds(left, right)
    print(f"[MANUAL] action={action}  L={left:+d}  R={right:+d}")
    return jsonify({'success': True, 'action': action, 'left': left, 'right': right})


@app.route('/api/estop', methods=['POST'])
def estop():
    """Immediately stop the motors."""
    if not PICO_IP:
        return jsonify({'success': False, 'error': 'No PICO_IP set'})
    threading.Thread(target=_send_stop_async, daemon=True).start()
    return jsonify({'success': True})

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    global conversation_history
    conversation_history = []
    return jsonify({'success': True})

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    question = data.get('question', '').strip()
    answer = data.get('answer', '').strip()
    if not question or not answer:
        return jsonify({'error': 'Both question and answer are required'}), 400
    new_id = add_qa_pair(question, answer)
    return jsonify({'success': True, 'id': new_id})

@app.route('/api/training-data', methods=['GET'])
def get_training_data():
    return jsonify(load_training_data())

@app.route('/api/training-data/<int:qa_id>', methods=['DELETE'])
def delete_training_data(qa_id):
    delete_qa_pair(qa_id)
    return jsonify({'success': True})

# ─────────────────────────────────────────────────────────
#  APP INITIALIZATION
# ─────────────────────────────────────────────────────────
def _emergency_stop():
    print("[SHUTDOWN] Sending stop to Pico before exit...")
    _send_stop_async()

if __name__ == '__main__':
    atexit.register(_emergency_stop)
    signal.signal(signal.SIGTERM, lambda *_: exit(0))

    try:
        init_storage()
        init_groq()
    except Exception as e:
        print(f"Init Config Warning: {e}")

    # Start the Tracking Threads automatically inside app.py
    if PHONE_IP and det_model:
        threading.Thread(target=frame_grabber, args=(VIDEO_URL,), daemon=True).start()
        time.sleep(2)
        threading.Thread(target=vision_loop, daemon=True).start()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cert_file = os.path.join(base_dir, 'cert.pem')
    key_file  = os.path.join(base_dir, 'key.pem')
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, ssl_context=(cert_file, key_file))
    else:
        print("Running in plain HTTP mode. (Web speech APIs may be blocked on mobile browsers)")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
