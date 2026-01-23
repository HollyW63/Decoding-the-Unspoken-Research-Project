import cv2
from deepface import DeepFace
import numpy as np
import torch
import pyaudio
from threading import Thread
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import csv
import datetime
import os
from collections import deque, Counter

# ---------- 1. UI Setup ----------
def get_color(emotion):
    """Return color based on emotion category."""
    if emotion in ['happy', 'neutral']:
        return (0, 255, 0) # Green (Positive)
    elif emotion in ['fear', 'sad', 'angry', 'disgust']:
        return (0, 0, 255) # Red (Negative)
    else:
        return (0, 165, 255) # Yellow (Complex)

# ---------- 2. Data Logging ----------
timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"session_data_{timestamp_str}.csv"

# Initialize CSV with headers
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Face_Emotion_Raw", "Face_Confidence", "Voice_Emotion_Raw", "Voice_Confidence"])

print(f"✅ Logging started: {csv_filename}")

# ---------- 3. Audio Model Setup ----------
model_name = "superb/wav2vec2-large-superb-er"
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
audio_model = AutoModelForAudioClassification.from_pretrained(model_name)

# Audio config
RATE = 16000
CHANNELS = 1
CHUNK = 1024
RECORD_SECONDS = 3
FORMAT = pyaudio.paInt16
audio_label = "neutral"
audio_conf = 0.0

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def audio_emotion_loop():
    """Background thread for real-time audio analysis."""
    global audio_label, audio_conf
    while True:
        try:
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            
            raw_data = np.concatenate(frames).astype(np.float32)

            # --- Volume Normalization ---
            # Boost audio signal if too quiet
            max_val = np.abs(raw_data).max()
            if max_val > 100:
                audio_data = raw_data / max_val 
            else:
                audio_data = raw_data / 32768.0 

            # Inference
            inputs = processor(torch.tensor(audio_data), sampling_rate=RATE, return_tensors="pt")
            with torch.no_grad():
                outputs = audio_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                pred_idx = torch.argmax(scores).item()
                
                # Update global vars
                audio_label = audio_model.config.id2label[pred_idx]
                audio_conf = scores[pred_idx].item()
        
        except Exception:
            continue

# Start Audio Thread
Thread(target=audio_emotion_loop, daemon=True).start()

# ---------- 4. Camera Loop & Processing ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

frame_skip = 5
frame_count = 0
face_results = []

# Smoothing: Queue to store recent emotions for stability
emotion_history = deque(maxlen=5) 
current_display_emotion = "neutral"
current_raw_emotion = "neutral"
current_face_conf = 0.0

print("🎥 System Running... Press 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame = cv2.flip(frame, 1) # Mirror effect

    # Run DeepFace every 'frame_skip' frames
    if frame_count % frame_skip == 0:
        try:
            face_results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        except Exception:
            face_results = []
        
        if len(face_results) > 0 and isinstance(face_results, list):
            main_face = face_results[0]
            current_raw_emotion = main_face['dominant_emotion']
            current_face_conf = main_face['emotion'][current_raw_emotion]
            
            # 1. Add to history
            emotion_history.append(current_raw_emotion)
            # 2. Get Mode (most frequent emotion)
            current_display_emotion = Counter(emotion_history).most_common(1)[0][0]

            # 3. Log Data to CSV
            now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([now, current_raw_emotion, f"{current_face_conf:.2f}", audio_label, f"{audio_conf:.2f}"])

    # Draw UI
    for face in face_results:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        
        # Display Stable Emotion
        color = get_color(current_display_emotion)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        label = f"{current_display_emotion.upper()} ({current_face_conf:.0f}%)"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display Audio Emotion
    tone_text = f"Voice: {audio_label} ({audio_conf*100:.0f}%)"
    cv2.putText(frame, tone_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Decoding the Unspoken - Research Build", frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()
print(f"🛑 Session Ended. Data saved: {csv_filename}")