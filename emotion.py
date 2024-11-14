from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import sqlite3
from datetime import datetime
import pickle
import json
import os
import time

class FaceEmotionTracker:
    def __init__(self, db_path='face_database.db', similarity_threshold=8, check_interval=20):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.check_interval = check_interval
        self.init_database()
        self.last_check_time = time.time()
        self.face_embeddings = {}
        self.new_face_detected = False
        self.known_face_detected = False
        self.face_status = "Unknown"

    def init_database(self):
        if os.path.exists(self.db_path):
            try:
                sqlite3.connect(self.db_path).close()
            except sqlite3.Error:
                os.remove(self.db_path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                emotion_history TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def get_face_embedding(self, face_img):
        try:
            result = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
            return result[0]['embedding'] if isinstance(result, list) else result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def is_face_similar(self, embedding1, embedding2):
        if embedding1 is None or embedding2 is None:
            return False
        try:
            distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
            return distance < self.similarity_threshold
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return False

    def store_new_face(self, face_id, face_img, emotion):
        embedding = self.get_face_embedding(face_img)
        if embedding is None:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        current_time = datetime.now()
        emotion_history = json.dumps([{'emotion': emotion, 'timestamp': current_time.isoformat()}])
        cursor.execute('''
            INSERT INTO faces (id, embedding, first_seen, last_seen, emotion_history)
            VALUES (?, ?, ?, ?, ?)
        ''', (face_id, pickle.dumps(embedding), current_time, current_time, emotion_history))
        conn.commit()
        conn.close()
        print("New face stored successfully.")
        self.new_face_detected = True
        self.face_status = "New face detected and stored!"

    def update_existing_face(self, face_id, emotion):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT emotion_history FROM faces WHERE id = ?', (face_id,))
        result = cursor.fetchone()
        if result:
            emotion_history = json.loads(result[0])
            emotion_history.append({'emotion': emotion, 'timestamp': datetime.now().isoformat()})
            cursor.execute('''
                UPDATE faces SET last_seen = ?, emotion_history = ?
                WHERE id = ?
            ''', (datetime.now(), json.dumps(emotion_history), face_id))
            self.known_face_detected = True
            self.face_status = "Known face detected!"
        conn.commit()
        conn.close()

    def process_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        self.new_face_detected = False
        self.known_face_detected = False
        if len(faces) == 0:
            self.face_status = "Unknown"
        current_time = time.time()
        should_check_database = (current_time - self.last_check_time) >= self.check_interval

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y+h, x:x+w]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emotion_text = f"{emotion} ({result[0]['emotion'][emotion]:.1f}%)"
            except Exception as e:
                emotion_text = "Error"
                print(f"Error analyzing emotion: {e}")
                continue

            face_id = hash(face_roi.tobytes())
            if face_id not in self.face_embeddings:
                if should_check_database:
                    embedding = self.get_face_embedding(face_roi)
                    if embedding is not None:
                        self.face_embeddings[face_id] = embedding
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute('SELECT id, embedding FROM faces')
                        is_new_face = True
                        for stored_face_id, stored_embedding_blob in cursor.fetchall():
                            stored_embedding = pickle.loads(stored_embedding_blob)
                            if self.is_face_similar(embedding, stored_embedding):
                                is_new_face = False
                                self.update_existing_face(stored_face_id, emotion)
                                break
                        if is_new_face:
                            self.store_new_face(face_id, face_roi, emotion)
                        conn.close()
                        self.last_check_time = current_time
            else:
                self.update_existing_face(face_id, emotion)
            color = (0, 255, 0) if self.known_face_detected else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, self.face_status, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def generate_frames(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()

    def get_current_status(self):
        return {
            "new_face": self.new_face_detected,
            "known_face": self.known_face_detected,
            "status": self.face_status
        }

app = Flask(__name__)
tracker = FaceEmotionTracker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(tracker.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_status')
def face_status():
    return jsonify(tracker.get_current_status())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
