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
FORWARD_SPEED  = 100
BACKWARD_SPEED = 90
TURN_SPEED     = 85

TURN_DEADZONE  = 100
DIST_DEADZONE  = 0.08
TARGET_HEIGHT_RATIO = 0.60

VIDEO_URL = f"http://{PHONE_IP}:8080/video"

if PHONE_IP and PICO_IP:
    print("Loading YOLOv8n models...")
    det_model  = YOLO('yolov8n.pt')
    pose_model = YOLO('yolov8n-pose.pt')
else:
    print("WARNING: Missing PHONE_IP or PICO_IP. Tracking will not function.")
    det_model = None
    pose_model = None

SHOULDER_L, SHOULDER_R = 5, 6
WRIST_L,    WRIST_R    = 9, 10

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
def turn_left():     send_speeds(-TURN_SPEED,       TURN_SPEED)
def turn_right():    send_speeds( TURN_SPEED,      -TURN_SPEED)
def stop():
    send_speeds(0, 0)
    threading.Thread(target=_send_stop_async, daemon=True).start()

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

def vision_loop():
    global output_frame, active_mode
    if not det_model:
        return

    last_left, last_right = 0, 0
    frames_since_cmd = 0
    pose_check_every = 5
    frame_counter = 0

    print("Vision loop started. Waiting for first frame...")
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()

        if active_mode != "follower":
            # Just quickly pause without doing tracking computation
            time.sleep(0.1)
            continue

        frame_counter += 1
        h, w, _ = frame.shape
        frame_cx = w // 2
        annotated = frame.copy()
        action_label = "SEARCHING..."
        label_color  = (200, 200, 200)
        next_left = next_right = 0

        hand_raised = False
        if frame_counter % pose_check_every == 0:
            pose_results = pose_model(frame, verbose=False)
            hand_raised  = is_hand_raised(pose_results)

        if hand_raised:
            stop()
            action_label = "STOP — Hand Raised"
            label_color  = (0, 0, 255)
            cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 180), -1)
            cv2.putText(annotated, "✋  " + action_label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            with stream_lock:
                output_frame = annotated.copy()
            continue

        det_results = det_model(frame, classes=[0], verbose=False)

        if len(det_results[0].boxes) == 0:
            stop()
            action_label = "NO HUMAN — STOPPED"
            label_color  = (0, 120, 255)
            cv2.putText(annotated, action_label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, label_color, 2)
            with stream_lock:
                output_frame = annotated.copy()
            continue

        box = det_results[0].boxes[0].xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 80), 2)

        box_h        = y2 - y1
        person_cx    = (x1 + x2) // 2
        horiz_offset = person_cx - frame_cx
        cur_ratio    = box_h / float(h)
        dist_error   = TARGET_HEIGHT_RATIO - cur_ratio

        approx_cm = max(1, int((TARGET_HEIGHT_RATIO / cur_ratio) * 10))

        cv2.line(annotated, (frame_cx, h // 2), (person_cx, h // 2), (0, 255, 80), 2)
        cv2.circle(annotated, (person_cx, (y1 + y2) // 2), 6, (0, 255, 80), -1)

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

        cv2.rectangle(annotated, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(annotated, action_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
        cv2.putText(annotated, f"L:{next_left:+d}  R:{next_right:+d} | ratio:{cur_ratio:.2f} | x_off:{horiz_offset:+d}px | ~{approx_cm}cm", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        changed = (abs(next_left - last_left) > 5 or abs(next_right - last_right) > 5)
        if changed or frames_since_cmd > 8:
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
    
    if new_mode in ["chatbot", "follower"]:
        old_mode = active_mode
        active_mode = new_mode
        print(f"[MODE SWITCH] Changed from {old_mode} to {active_mode}")
        
        if old_mode == "follower" and active_mode != "follower":
            # Force massive wheels stop since tracking loop is now sleeping
            print("[SAFETY] Force stopping wheels because exiting Follower mode...")
            stop()
            
        return jsonify({'success': True, 'mode': active_mode})
    return jsonify({'error': 'Invalid mode'}), 400

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
