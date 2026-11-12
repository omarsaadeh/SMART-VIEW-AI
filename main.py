import cv2
import socket
import time
import queue
import threading
import torch
from ultralytics import YOLO

# =============================
# CUDA & PERFORMANCE CONFIG
# =============================
print("CUDA Available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# GLOBAL FLAGS & THREAD CONTROL
# =============================
running = True
speech_queue = queue.Queue(maxsize=1)
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
mode_lock = threading.Lock()

# =============================
# SOCKET SETUP
# =============================
RASP_IP = "172.20.10.8"
PORT = 5014
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connected = False
try:
    s.connect((RASP_IP, PORT))
    connected = True
    print("Connected to Raspberry Pi successfully.")
except Exception as e:
    print("Socket connection failed:", e)

# =============================
# NETWORK THREAD
# =============================
def network_sender_thread():
    global connected, running
    while running:
        try:
            message = speech_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if message is None:
            break
        if connected:
            try:
                s.sendall((message + "\n").encode())
            except Exception as e:
                print("Network send error:", e)
                connected = False

send_thread = threading.Thread(target=network_sender_thread, daemon=True)
send_thread.start()

# =============================
# LOAD MODELS
# =============================
print("Loading General Model (YOLOv8x)...")
model_general = YOLO("yolov8x.pt")
model_general.to(device)

print("Loading Money Model...")
try:
    model_money = YOLO("best.pt")
    model_money.to(device)
    print("Money model loaded successfully!")
    print("Classes:", model_money.names)
except Exception as e:
    print("Error loading best.pt:", e)
    model_money = model_general

# =============================
# PRIORITY DICTIONARY
# =============================
PRIORITY_DICT = {
    'person': 1, 'car': 1, 'motorcycle': 1, 'bus': 1, 'truck': 1, 'bicycle': 1, 
    'stop sign': 1, 'traffic light': 1, 'fire hydrant': 1,
    'chair': 2, 'couch': 2, 'bed': 2, 'dining table': 2, 'bench': 2, 'door': 2
}
DEFAULT_PRIORITY = 3

# =============================
# CURRENT MODE
# =============================
current_mode = 1

# =============================
# CAMERA STREAM
# =============================
url = "http://172.20.10.2:8080/video"
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# =============================
# TRACKING MEMORY & DEBOUNCING
# =============================
track_history = {}
global_cooldowns = {}
TRACK_TIMEOUT = 5.0
SPEECH_COOLDOWN = 3.5
last_spoken_time = 0.0
REQUIRED_FRAMES = 5

# =============================
# CAMERA THREAD
# =============================
def camera_thread():
    global running
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
        frame_queue.put(frame)

# =============================
# YOLO THREAD
# =============================
def yolo_thread():
    global running, current_mode, last_spoken_time
    frame_count = 0
    local_mode = current_mode
    prev_time = time.time()

    while running:
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        if local_mode != current_mode:
            track_history.clear()
            global_cooldowns.clear()
            local_mode = current_mode

        h, w, _ = frame.shape
        mid_x = w // 2

        with mode_lock:
            if current_mode == 1:
                active_model = model_general
                MODEL_IMG_SIZE = 160
                CONFIDENCE_THRESHOLD = 0.60  # رفعنا الثقة لتجنب الأشياء البعيدة والغير واضحة
                mode_text = "MODE: GENERAL (YOLOv8x)"
                mode_color = (0, 255, 0)
                
                # حد المساحة للموديل العام (إذا البوكس أصغر من هول البكسلات يعني بعيد كتير وبطنشو)
                MIN_AREA_THRESHOLD = 18000 
            else:
                active_model = model_money
                MODEL_IMG_SIZE = 480
                CONFIDENCE_THRESHOLD = 0.60
                mode_text = "MODE: MONEY DETECTION"
                mode_color = (0, 0, 255)
                
                # بمود العملة بدنا نقرا كل شي حتى لو بعيد شوي كرمال المستخدم يشوف شو معه كاش
                MIN_AREA_THRESHOLD = 4000 

        try:
            results = active_model.track(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=MODEL_IMG_SIZE,
                device=device,
                persist=True,
                tracker="botsort.yaml",
                verbose=False
            )
        except Exception as e:
            print("YOLO ERROR:", e)
            continue

        now = time.time()
        best_target_phrase = None
        best_area = 0
        best_priority = 999

        for track_id in list(track_history.keys()):
            if now - track_history[track_id]['last_seen'] > TRACK_TIMEOUT:
                del track_history[track_id]

        for label in list(global_cooldowns.keys()):
            if now - global_cooldowns[label] > TRACK_TIMEOUT:
                del global_cooldowns[label]

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            
            for box, cls, track_id_tensor in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.id):
                b = box.cpu().numpy().astype(int)
                label = active_model.names[int(cls)]
                track_id = int(track_id_tensor.item())

                # حساب مساحة الجسم (Bounding Box Area)
                area = (b[2] - b[0]) * (b[3] - b[1])

                # الفيلتر السحري: إذا الجسم أصغر من الحد المسموح، منرسم البوكس بس منعملّه skip من النطق
                if area < MIN_AREA_THRESHOLD:
                    # منرسمه بخط رفيع على الشاشة للتجربة بس ما بنطقه
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 1)
                    cv2.putText(frame, f"Too Far ({area})", (b[0], b[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    continue

                if track_id not in track_history:
                    track_history[track_id] = {'counts': {}, 'last_seen': now, 'spoken': False}

                track_history[track_id]['last_seen'] = now
                track_history[track_id]['counts'][label] = track_history[track_id]['counts'].get(label, 0) + 1

                best_label_for_track = max(track_history[track_id]['counts'], key=track_history[track_id]['counts'].get)
                label_count = track_history[track_id]['counts'][best_label_for_track]

                cx = (b[0] + b[2]) // 2
                position = "left" if cx < mid_x else "right"
                speech_phrase = f"{best_label_for_track} on the {position}"

                # رسم الأجسام القريبة والمؤكدة بلون الموديل
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), mode_color, 2)
                display_tag = f"ID:{track_id} {best_label_for_track} ({label_count}/{REQUIRED_FRAMES})"
                cv2.putText(frame, display_tag, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)

                # =============================
                # PRIORITY & SMART SPEECH SYSTEM
                # =============================
                if not track_history[track_id]['spoken'] and label_count >= REQUIRED_FRAMES:
                    if (best_label_for_track not in global_cooldowns or (now - global_cooldowns[best_label_for_track] > TRACK_TIMEOUT)):
                        
                        current_item_priority = PRIORITY_DICT.get(best_label_for_track, DEFAULT_PRIORITY) if current_mode == 1 else 1

                        if current_item_priority < best_priority or (current_item_priority == best_priority and area > best_area):
                            best_priority = current_item_priority
                            best_area = area
                            best_target_phrase = (speech_phrase, track_id, best_label_for_track)

        if best_target_phrase and (now - last_spoken_time >= SPEECH_COOLDOWN):
            phrase_to_say, target_id, target_label = best_target_phrase
            try:
                speech_queue.put(phrase_to_say, block=False)
                print(f'ENQUEUED: "{phrase_to_say}" (Priority: {best_priority}, Area: {best_area})')
                
                track_history[target_id]['spoken'] = True
                global_cooldowns[target_label] = now
                last_spoken_time = now
            except queue.Full:
                pass

        cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 0, 0), 2)
        cv2.putText(frame, "LEFT ZONE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "RIGHT ZONE", (mid_x + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if result_queue.full():
            try:
                result_queue.get_nowait()
            except:
                pass
        result_queue.put(frame)

cam_thread = threading.Thread(target=camera_thread, daemon=True)
detect_thread = threading.Thread(target=yolo_thread, daemon=True)
cam_thread.start()
detect_thread.start()

print("System Ready")
print("Press 1 -> General Mode (YOLOv8x)")
print("Press 2 -> Money Mode")
print("Press Q -> Quit")

while True:
    try:
        frame = result_queue.get(timeout=0.1)
    except queue.Empty:
        continue
    
    cv2.imshow("SMART VIEW AI", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        with mode_lock:
            if current_mode != 1:
                current_mode = 1
                print("Switched to GENERAL mode")
    elif key == ord('2'):
        with mode_lock:
            if current_mode != 2:
                current_mode = 2
                print("Switched to MONEY mode")
    elif key == ord('q'):
        print("Shutting down...")
        break

running = False
time.sleep(0.5)
cap.release()
speech_queue.put(None)
send_thread.join()
s.close()
cv2.destroyAllWindows()
print("System exited cleanly.")