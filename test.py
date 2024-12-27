import cv2
import requests
import time
import os
import threading
from datetime import datetime

# Telegram
import telebot

# YOLOv8
from ultralytics import YOLO

# ================== TELEGRAM BOT CONFIG ==================
BOT_TOKEN = '7812223667:AAGsx3Kl7W6ZfxnPdXx4rHtpU8SY9uJpWHE'
CHAT_ID = '828896533'
bot = telebot.TeleBot(BOT_TOKEN)

# ================== YOLOv8 MODEL CONFIG ==================
MODEL_PATH = 'train3/weights/best.pt'  # Your YOLOv8 weights
CONFIDENCE_THRESHOLD = 0.8
MONKEY_CLASS_ID = 0   # If your "monkey" class is index 0, adjust if needed

model = YOLO(MODEL_PATH)

# ================== CAMERA / STREAM CONFIG ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# The Render endpoint (adjust to your actual domain/route)
UPLOAD_URL = "https://monkeydetection-web.onrender.com/upload_frame"

# ================== VIDEO RECORD CONFIG ==================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
record_duration = 5  # how many seconds to record when monkey is detected
frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# To prevent multiple recordings at the same time
detection_thread = None

def send_frame_to_render(frame):
    """
    Encodes `frame` to JPEG and sends to the Render server for streaming.
    """
    _, encoded_image = cv2.imencode('.jpg', frame)
    files = {'frame': ('frame.jpg', encoded_image.tobytes(), 'image/jpeg')}
    try:
        resp = requests.post(UPLOAD_URL, files=files, timeout=1)
        print("POST:", resp.status_code, resp.text)
    except Exception as e:
        print("Error uploading frame:", e)

def play_alarm():
    """Play alarm sound (optional)."""
    try:
        # Example for macOS:
        # os.system('afplay /path/to/alarm.wav')
        # Example for Linux:
        # os.system('aplay /path/to/alarm.wav')
        pass
    except Exception as e:
        print(f"Error playing alarm: {e}")

def send_alert_telegram(video_path):
    """
    Send an alert message and the recorded video to Telegram.
    """
    try:
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_message = f"🚨 Monkey detected!\nTimestamp: {timestamp}"
        bot.send_message(CHAT_ID, alert_message)

        with open(video_path, 'rb') as video:
            bot.send_video(CHAT_ID, video)

        print(f"Video sent successfully: {video_path}")
        os.remove(video_path)  # Remove the file after sending
    except Exception as e:
        print(f"Error sending video to Telegram: {e}")

def record_and_alert(cap, frame_rate):
    """
    Records 'record_duration' seconds from the webcam, saves to a file,
    then sends it to Telegram, and plays an alarm.
    Runs in a background thread.
    """
    # 1. Record video
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_name = f"monkey_event_{timestamp}.mp4"
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(video_name, fourcc, frame_rate, (frame_width, frame_height))
    start_time = time.time()

    while time.time() - start_time < record_duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()

    # 2. Send to Telegram
    send_alert_telegram(video_name)

    # 3. (Optional) Play alarm
    play_alarm()

def main():
    global detection_thread

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # ============ 1. RUN YOLOv8 DETECTION =============
        results_list = model.predict(frame, conf=CONFIDENCE_THRESHOLD)
        results = results_list[0]  # one result for this single frame
        monkey_detected = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]

            if cls_id == MONKEY_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
                monkey_detected = True
                print("Detected a monkey!")

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"Monkey {conf:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ============ 2. STREAM THE FRAME TO RENDER =============
        send_frame_to_render(frame)

        # ============ 3. IF MONKEY DETECTED, RECORD + SEND TO TELEGRAM =============
        # Only start a new thread if the previous one is finished
        if monkey_detected and (detection_thread is None or not detection_thread.is_alive()):
            detection_thread = threading.Thread(target=record_and_alert, args=(cap, frame_rate))
            detection_thread.start()

        # ============ 4. SHOW LOCALLY (Optional) =============
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Slight delay for ~30 FPS
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()