from flask import Flask, render_template, Response, stream_with_context, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import sqlite3
from datetime import datetime
import pickle
import json
import os
import time
from queue import Queue
import threading
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Create a queue for face events
face_event_queue = Queue()

# Ensure the static and templates directories exist
Path(app.static_folder).mkdir(parents=True, exist_ok=True)
Path(app.template_folder).mkdir(parents=True, exist_ok=True)

class FaceEmotionTracker:
    def __init__(self, db_path='face_database.db', similarity_threshold=8, check_interval=20):
        """Initialize the face emotion tracker with given parameters."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.check_interval = check_interval
        self.init_database()
        self.last_check_time = time.time()
        self.face_embeddings = {}
        self.video_capture = None

    def init_database(self):
        """Initialize SQLite database for storing face data."""
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
        """Generate face embedding using DeepFace."""
        try:
            result = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
            return result[0]['embedding'] if isinstance(result, list) else result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def is_face_similar(self, embedding1, embedding2):
        """Compare two face embeddings for similarity."""
        if embedding1 is None or embedding2 is None:
            return False
        try:
            distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
            return distance < self.similarity_threshold
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return False

    def store_new_face(self, face_id, face_img, emotion):
        """Store a new face in the database."""
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
        face_event_queue.put('new_face')

    def update_existing_face(self, face_id, emotion):
        """Update emotion history for an existing face."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT emotion_history FROM faces WHERE id = ?', (face_id,))
        result = cursor.fetchone()
        
        if result:
            emotion_history = json.loads(result[0])
            emotion_history.append({
                'emotion': emotion,
                'timestamp': datetime.now().isoformat()
            })
            cursor.execute('''
                UPDATE faces SET last_seen = ?, emotion_history = ?
                WHERE id = ?
            ''', (datetime.now(), json.dumps(emotion_history), face_id))
        
        conn.commit()
        conn.close()

    def process_frame(self, frame):
        """Process a video frame for face detection and emotion analysis."""
        if frame is None:
            return None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
                            print("New face detected!")
                            self.store_new_face(face_id, face_roi, emotion)
                        
                        conn.close()
                        self.last_check_time = current_time
            else:
                self.update_existing_face(face_id, emotion)

            # Draw rectangle and emotion text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, emotion_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    def generate_frames(self):
        """Generator function for video streaming."""
        self.video_capture = cv2.VideoCapture(0)
        
        try:
            while True:
                success, frame = self.video_capture.read()
                if not success:
                    break
                
                processed_frame = self.process_frame(frame)
                if processed_frame is None:
                    continue
                
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        finally:
            if self.video_capture:
                self.video_capture.release()

# Flask routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    tracker = FaceEmotionTracker()
    return Response(
        tracker.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/face-events')
def face_events():
    """Server-sent events route for face detection notifications."""
    def event_stream():
        while True:
            event = face_event_queue.get()
            yield f"data: {event}\n\n"
            time.sleep(0.1)

    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream'
    )

@app.route('/start_video')
def start_video():
    """Start video capture."""
    tracker = FaceEmotionTracker()
    if tracker.video_capture is None:
        tracker.video_capture = cv2.VideoCapture(0)
    return jsonify({'status': 'success'})

@app.route('/stop_video')
def stop_video():
    """Stop video capture."""
    tracker = FaceEmotionTracker()
    if tracker.video_capture:
        tracker.video_capture.release()
        tracker.video_capture = None
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # Ensure the database directory exists
    db_dir = os.path.dirname('face_database.db')
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Run the Flask app
    app.run(debug=True, threaded=True)