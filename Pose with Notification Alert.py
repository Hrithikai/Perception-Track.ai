import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import csv
from datetime import datetime
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =========================
# OPTIMIZED CONFIGURATION
# =========================
# Image quality settings - OPTIMIZED FOR SPEED
ALERT_IMAGE_WIDTH = 640
ALERT_IMAGE_HEIGHT = 480
ALERT_IMAGE_QUALITY = 65
MAX_IMAGE_SIZE_BYTES = 500 * 1024

# Network timeouts - AGGRESSIVE
TELEGRAM_TIMEOUT = 3
EMAIL_TIMEOUT = 8

# Processing optimization
YOLO_CONF_THRESHOLD = 0.5
YOLO_IMG_SIZE = 640

# =========================
# NOTIFICATION CREDENTIALS
# =========================
BOT_TOKEN = "8331904126:AAGjeeo7hWO-CHAsYAhQ7ZctRSNqpUwBAo4"
CHAT_ID = "1490935977"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

SENDER_ACCOUNTS = [
    {
        "email": "boostup1947@gmail.com",
        "password": "bkcl hcuv gcjf gslz"
    }
]

RECEIVER_EMAILS = [
    "hrithiksai007@gmail.com",
    "hrithiksai.n2021@vitstudent.ac.in",
    "sweta.b@vit.ac.in",
    "senthilkumar.t@vit.ac.in"
]

# =========================
# THREAD-SAFE ASYNC LOOP
# =========================
class AsyncEventLoop:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_loop(cls):
        with cls._lock:
            if cls._instance is None:
                try:
                    cls._instance = asyncio.get_event_loop()
                    if cls._instance.is_closed():
                        raise RuntimeError("Loop is closed")
                except RuntimeError:
                    cls._instance = asyncio.new_event_loop()
                    asyncio.set_event_loop(cls._instance)
            return cls._instance

# =========================
# ULTRA-FAST IMAGE OPTIMIZER
# =========================
def optimize_image_ultra_fast(image_path: str) -> str:
    """Aggressive image optimization for fastest transmission."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not read image: {image_path}")
            return image_path
        
        img_resized = cv2.resize(img, (ALERT_IMAGE_WIDTH, ALERT_IMAGE_HEIGHT), 
                                 interpolation=cv2.INTER_AREA)
        
        optimized_path = image_path.replace('.jpg', '_opt.jpg')
        
        quality = ALERT_IMAGE_QUALITY
        while quality > 30:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            cv2.imwrite(optimized_path, img_resized, encode_params)
            
            if os.path.exists(optimized_path):
                file_size = os.path.getsize(optimized_path)
                if file_size <= MAX_IMAGE_SIZE_BYTES:
                    print(f"✅ Image optimized: {file_size/1024:.1f}KB (quality={quality})")
                    return optimized_path
            
            quality -= 10
        
        cv2.imwrite(optimized_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 30])
        print(f"✅ Image optimized to minimum quality")
        return optimized_path
        
    except Exception as e:
        print(f"❌ Image optimization error: {e}")
        return image_path

# =========================
# ULTRA-FAST TELEGRAM
# =========================
async def send_telegram_message_ultra(session, message: str):
    """Send Telegram message with minimal timeout."""
    try:
        url = f"{BASE_URL}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        
        timeout = aiohttp.ClientTimeout(total=TELEGRAM_TIMEOUT)
        async with session.post(url, data=payload, timeout=timeout) as resp:
            if resp.status == 200:
                print("✅ Telegram message sent")
                return await resp.json()
            else:
                print(f"⚠️ Telegram message status: {resp.status}")
                return None
                
    except asyncio.TimeoutError:
        print("⚠️ Telegram message timeout - continuing")
        return None
    except Exception as e:
        print(f"⚠️ Telegram message error: {str(e)[:50]}")
        return None

async def send_telegram_photo_ultra(session, image_path: str, caption: str):
    """Send optimized photo ultra-fast."""
    optimized_path = None
    try:
        optimized_path = optimize_image_ultra_fast(image_path)
        
        if not os.path.exists(optimized_path):
            print(f"⚠️ Optimized image not found: {optimized_path}")
            return None
        
        url = f"{BASE_URL}/sendPhoto"
        
        with open(optimized_path, "rb") as photo:
            form = aiohttp.FormData()
            form.add_field("chat_id", CHAT_ID)
            form.add_field("caption", caption[:200])
            form.add_field("photo", photo, 
                          filename=os.path.basename(optimized_path),
                          content_type="image/jpeg")
            
            timeout = aiohttp.ClientTimeout(total=TELEGRAM_TIMEOUT)
            async with session.post(url, data=form, timeout=timeout) as resp:
                if resp.status == 200:
                    print("✅ Telegram photo sent")
                    return await resp.json()
                else:
                    print(f"⚠️ Telegram photo status: {resp.status}")
                    return None
                    
    except asyncio.TimeoutError:
        print("⚠️ Telegram photo timeout - continuing")
        return None
    except Exception as e:
        print(f"⚠️ Telegram photo error: {str(e)[:50]}")
        return None
    finally:
        if optimized_path and optimized_path != image_path:
            try:
                if os.path.exists(optimized_path):
                    os.remove(optimized_path)
            except:
                pass

async def send_telegram_alert_ultra(message: str, image_path: str = None):
    """Send complete Telegram alert ultra-fast."""
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=TELEGRAM_TIMEOUT * 2)
    
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            tasks.append(send_telegram_message_ultra(session, message))
            
            if image_path and os.path.exists(image_path):
                tasks.append(send_telegram_photo_ultra(session, image_path, message[:100]))
            
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 
                                  timeout=TELEGRAM_TIMEOUT * 2)
            
    except asyncio.TimeoutError:
        print("⚠️ Telegram alert timeout - continuing")
    except Exception as e:
        print(f"⚠️ Telegram alert error: {str(e)[:50]}")

# =========================
# ULTRA-FAST EMAIL
# =========================
def send_email_ultra(subject: str, body: str, image_path: str = None):
    """Send email ultra-fast with optimization."""
    optimized_path = None
    
    for sender in SENDER_ACCOUNTS:
        try:
            msg = MIMEMultipart()
            msg['From'] = sender["email"]
            msg['To'] = ", ".join(RECEIVER_EMAILS)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, "plain"))
            
            if image_path and os.path.exists(image_path):
                optimized_path = optimize_image_ultra_fast(image_path)
                
                if os.path.exists(optimized_path):
                    with open(optimized_path, "rb") as file:
                        mime = MIMEBase("application", "octet-stream")
                        mime.set_payload(file.read())
                        encoders.encode_base64(mime)
                        mime.add_header("Content-Disposition", 
                                       f"attachment; filename={os.path.basename(optimized_path)}")
                        msg.attach(mime)
            
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=EMAIL_TIMEOUT)
            server.starttls()
            server.login(sender["email"], sender["password"])
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email sent from {sender['email']}")
            return True
            
        except smtplib.SMTPException as e:
            print(f"⚠️ SMTP error from {sender['email']}: {str(e)[:50]}")
            continue
        except Exception as e:
            print(f"⚠️ Email error from {sender['email']}: {str(e)[:50]}")
            continue
        finally:
            if optimized_path and optimized_path != image_path:
                try:
                    if os.path.exists(optimized_path):
                        os.remove(optimized_path)
                except:
                    pass
    
    return False

# =========================
# UNIFIED ALERT SYSTEM
# =========================
def send_alert_ultra_fast(message: str, image_path: str = None):
    """Send alerts via Telegram and Email in parallel."""
    def send_telegram_task():
        try:
            loop = AsyncEventLoop.get_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    send_telegram_alert_ultra(message, image_path),
                    loop
                )
            else:
                loop.run_until_complete(send_telegram_alert_ultra(message, image_path))
        except Exception as e:
            print(f"⚠️ Telegram task error: {str(e)[:50]}")
    
    def send_email_task():
        try:
            send_email_ultra("🚨 Pose Alert", message, image_path)
        except Exception as e:
            print(f"⚠️ Email task error: {str(e)[:50]}")
    
    executor = ThreadPoolExecutor(max_workers=2)
    executor.submit(send_telegram_task)
    executor.submit(send_email_task)
    
    print("⚡ Alerts dispatched in parallel")

# =========================
# FAST CSV MONITOR
# =========================
class UltraFastCSVMonitor:
    def __init__(self, csv_file_path, check_interval=2):
        self.csv_file = csv_file_path
        self.check_interval = check_interval
        self.last_processed_index = 0
        self.sent_alerts = set()
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"📊 CSV monitoring started: {self.csv_file}")
        
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3)
        print("✅ CSV monitoring stopped")
        
    def _monitor_loop(self):
        while self.monitoring:
            try:
                self._check_new_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"⚠️ CSV monitor error: {str(e)[:50]}")
                time.sleep(3)
                
    def _check_new_alerts(self):
        if not os.path.exists(self.csv_file):
            return
            
        try:
            df = pd.read_csv(self.csv_file)
            
            if len(df) == 0:
                return
            
            new_alerts = df[
                (df['long_term_alert'] == True) & 
                (df.index >= self.last_processed_index)
            ]
            
            for index, row in new_alerts.iterrows():
                alert_id = f"{row['timestamp']}_{row['alert_count']}"
                
                if alert_id not in self.sent_alerts:
                    self._process_alert(row, alert_id)
                    self.sent_alerts.add(alert_id)
            
            if len(df) > 0:
                self.last_processed_index = len(df) - 1
                
        except Exception as e:
            print(f"⚠️ CSV read error: {str(e)[:50]}")
            
    def _process_alert(self, row, alert_id):
        try:
            timestamp = row['datetime']
            duration = row['alert_duration']
            alert_count = row['alert_count']
            confidence = row['confidence']
            person_id = row['person_id']
            
            message = f"""🚨 ALERT #{alert_count}

👤 Person ID: {person_id}
⏰ Time: {timestamp}
⏱️ Duration: {duration:.1f}s
🎯 Confidence: {confidence:.2f}

⚠️ Person lying down detected!"""

            image_path = self._find_alert_image(timestamp, alert_count)
            send_alert_ultra_fast(message, image_path)
            print(f"⚡ Alert processed: {alert_id}")
            
        except Exception as e:
            print(f"⚠️ Alert process error: {str(e)[:50]}")

    def _find_alert_image(self, timestamp, alert_count):
        try:
            alerts_dir = os.path.join(os.path.dirname(self.csv_file), "alerts")
            if not os.path.exists(alerts_dir):
                return None
            
            files = os.listdir(alerts_dir)
            
            for filename in files:
                if f"alert_{alert_count}_" in filename and filename.endswith('.jpg'):
                    return os.path.join(alerts_dir, filename)
            
            jpg_files = [f for f in files if f.endswith('.jpg') and 'alert' in f.lower()]
            if jpg_files:
                jpg_files.sort(reverse=True)
                return os.path.join(alerts_dir, jpg_files[0])
                
            return None
            
        except Exception as e:
            print(f"⚠️ Image find error: {str(e)[:30]}")
            return None

# =========================
# OUTPUT DIRECTORIES
# =========================
OUTPUT_DIR = "pose_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(OUTPUT_DIR, "pose_analysis.csv")
ALERT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "alerts")
os.makedirs(ALERT_IMAGES_DIR, exist_ok=True)

VIDEO_PATH = r"D:\Perception Track.ai\dataset\VID_20251007_180327395~2.mp4"

# =========================
# LOAD YOLO MODEL
# =========================
try:
    pose_model = YOLO("yolov8s-pose.pt")
    print("✅ YOLO Pose model loaded")
except Exception as e:
    print(f"❌ YOLO model error: {e}")
    exit(1)

# =========================
# COMPLETE SKELETON DRAWING
# =========================
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def draw_complete_skeleton(frame, keypoints, pose_label):
    """Draw ALL 17 keypoints and complete skeleton connections."""
    if keypoints is None or len(keypoints) < 17:
        return frame
    
    if pose_label == 'lying_down':
        skeleton_color = (0, 255, 255)
        keypoint_color = (0, 0, 255)
    elif pose_label == 'sitting':
        skeleton_color = (255, 255, 0)
        keypoint_color = (255, 0, 0)
    else:
        skeleton_color = (0, 255, 0)
        keypoint_color = (0, 255, 0)
    
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and
            len(keypoints[start_idx]) >= 2 and len(keypoints[end_idx]) >= 2):
            
            x1, y1 = keypoints[start_idx][:2]
            x2, y2 = keypoints[end_idx][:2]
            
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.line(frame, pt1, pt2, skeleton_color, 2, cv2.LINE_AA)
    
    for i, kp in enumerate(keypoints):
        if len(kp) >= 2:
            x, y = kp[:2]
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 4, keypoint_color, -1, cv2.LINE_AA)
    
    return frame

# =========================
# MULTI-PERSON TRACKER
# =========================
class PersonTracker:
    """Track individual person with unique ID and bounding box."""
    def __init__(self, person_id, initial_bbox, initial_keypoints):
        self.id = person_id
        self.bbox = initial_bbox  # [x1, y1, x2, y2]
        self.keypoints = initial_keypoints
        self.pose = "standing"
        self.confidence = 0.0
        self.features = None
        self.last_seen = time.time()
        self.consecutive_lying_frames = 0
        self.lying_start_time = None
        self.alert_triggered = False
        self.alert_count = 0
        self.last_alert_time = 0
        
    def update(self, bbox, keypoints, pose, confidence, features):
        """Update tracker with new detection."""
        self.bbox = bbox
        self.keypoints = keypoints
        self.pose = pose
        self.confidence = confidence
        self.features = features
        self.last_seen = time.time()
    
    def is_active(self, current_time, timeout=2.0):
        """Check if person is still being tracked."""
        return (current_time - self.last_seen) < timeout

class MultiPersonTracker:
    """Manage multiple person trackers."""
    def __init__(self):
        self.trackers = {}
        self.next_id = 1
        self.iou_threshold = 0.3
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update(self, detections, current_time):
        """
        Update trackers with new detections.
        detections: list of (bbox, keypoints, pose, confidence, features)
        """
        # Match detections to existing trackers
        matched = set()
        
        for bbox, keypoints, pose, confidence, features in detections:
            best_match_id = None
            best_iou = self.iou_threshold
            
            # Find best matching tracker
            for tracker_id, tracker in self.trackers.items():
                if not tracker.is_active(current_time):
                    continue
                    
                iou = self.calculate_iou(bbox, tracker.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = tracker_id
            
            # Update existing tracker or create new one
            if best_match_id is not None:
                self.trackers[best_match_id].update(bbox, keypoints, pose, confidence, features)
                matched.add(best_match_id)
            else:
                # Create new tracker
                new_id = self.next_id
                self.next_id += 1
                self.trackers[new_id] = PersonTracker(new_id, bbox, keypoints)
                self.trackers[new_id].update(bbox, keypoints, pose, confidence, features)
                matched.add(new_id)
                print(f"👤 New person detected: ID {new_id}")
        
        # Remove inactive trackers
        inactive = [tid for tid, tracker in self.trackers.items() 
                   if not tracker.is_active(current_time)]
        for tid in inactive:
            del self.trackers[tid]
        
        return list(matched)
    
    def get_active_trackers(self, current_time):
        """Get all active trackers."""
        return {tid: tracker for tid, tracker in self.trackers.items() 
                if tracker.is_active(current_time)}

# =========================
# POSE CLASSIFICATION
# =========================
def classify_pose_accurate(features, previous_pose="standing"):
    """Improved pose classification with better accuracy."""
    torso = features['torso_angle']
    knee = features['avg_knee_angle']
    hip = features['avg_hip_angle']
    
    scores = {'standing': 0.0, 'sitting': 0.0, 'lying_down': 0.0}
    
    if torso >= 50:
        lying_score = 0.65
        if torso >= 65: lying_score = 0.80
        if torso >= 80: lying_score = 0.95
        scores['lying_down'] = lying_score
    
    elif knee < 145:
        sitting_score = 0.60
        if 70 <= knee <= 135: sitting_score += 0.20
        if 85 <= hip <= 145: sitting_score += 0.15
        scores['sitting'] = min(sitting_score, 1.0)
    
    elif torso <= 40 and knee >= 150:
        standing_score = 0.70
        if torso <= 25: standing_score += 0.20
        if knee >= 165: standing_score += 0.10
        scores['standing'] = min(standing_score, 1.0)
    
    else:
        if previous_pose in scores:
            scores[previous_pose] = 0.35
    
    best_pose = max(scores, key=scores.get)
    best_confidence = scores[best_pose]
    
    if best_confidence < 0.45:
        return "uncertain", scores, previous_pose
    
    return best_pose, scores, best_pose

# =========================
# MULTI-PERSON ALERT SYSTEM
# =========================
class MultiPersonAlertSystem:
    """Alert system for multiple people."""
    def __init__(self, initial_threshold=10, max_alerts=3, cooldown=10):
        self.initial_threshold = initial_threshold
        self.max_alerts = max_alerts
        self.cooldown = cooldown
        self.min_consecutive_frames = 5
        self.global_alert_count = 0
        
    def update_person(self, tracker, current_time):
        """Update alert state for a specific person."""
        if tracker.pose == 'lying_down':
            tracker.consecutive_lying_frames += 1
            
            if tracker.consecutive_lying_frames >= self.min_consecutive_frames:
                if tracker.lying_start_time is None:
                    tracker.lying_start_time = current_time
                    print(f"⏱️ Person {tracker.id}: Lying timer started")
                
                lying_duration = current_time - tracker.lying_start_time
                
                # First alert
                if (lying_duration >= self.initial_threshold and 
                    not tracker.alert_triggered and 
                    tracker.alert_count < self.max_alerts):
                    
                    tracker.alert_triggered = True
                    tracker.alert_count += 1
                    tracker.last_alert_time = current_time
                    self.global_alert_count += 1
                    return True, lying_duration, tracker.alert_count, tracker.id
                
                # Subsequent alerts
                elif (tracker.alert_triggered and 
                      current_time - tracker.last_alert_time >= self.cooldown and
                      tracker.alert_count < self.max_alerts):
                    
                    tracker.alert_count += 1
                    tracker.last_alert_time = current_time
                    self.global_alert_count += 1
                    return True, lying_duration, tracker.alert_count, tracker.id
                    
            return False, 0, tracker.alert_count, tracker.id
            
        else:
            # Reset with hysteresis
            if tracker.consecutive_lying_frames > 0:
                tracker.consecutive_lying_frames = max(0, tracker.consecutive_lying_frames - 3)
                if tracker.consecutive_lying_frames == 0:
                    tracker.lying_start_time = None
                    tracker.alert_triggered = False
            return False, 0, tracker.alert_count, tracker.id

# =========================
# UTILITY FUNCTIONS
# =========================
class HumanValidator:
    """Validate human presence in keypoints."""
    def __init__(self):
        self.required_keypoints = [5, 6, 11, 12]
        self.min_keypoints_visible = 3
        
    def validate_human_presence(self, keypoints):
        if keypoints is None or len(keypoints) < 17:
            return False
        
        valid_count = 0
        for i in self.required_keypoints:
            if (i < len(keypoints) and len(keypoints[i]) >= 2 and 
                keypoints[i][0] > 5 and keypoints[i][1] > 5):
                valid_count += 1
        
        return valid_count >= self.min_keypoints_visible

def extract_pose_features(keypoints):
    """Extract pose features from keypoints."""
    try:
        keypoints_list = []
        indices = [5, 6, 11, 12, 13, 14, 15, 16]
        
        for i in indices:
            if i < len(keypoints) and len(keypoints[i]) >= 2:
                keypoints_list.append(keypoints[i][:2])
            else:
                keypoints_list.append(np.array([0, 0]))
        
        l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle = keypoints_list
        
        shoulder_mid = (l_shoulder + r_shoulder) / 2
        hip_mid = (l_hip + r_hip) / 2
        
        torso_angle = calculate_torso_angle(shoulder_mid, hip_mid)
        
        left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        knee_angles = [ang for ang in [left_knee_angle, right_knee_angle] if ang < 180]
        avg_knee_angle = np.mean(knee_angles) if knee_angles else 160.0
        
        left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        hip_angles = [ang for ang in [left_hip_angle, right_hip_angle] if ang < 180]
        avg_hip_angle = np.mean(hip_angles) if hip_angles else 140.0
        
        return {
            'torso_angle': torso_angle,
            'avg_knee_angle': avg_knee_angle,
            'avg_hip_angle': avg_hip_angle
        }
    except Exception as e:
        return None

def calculate_torso_angle(shoulder_mid, hip_mid):
    """Calculate torso angle from vertical."""
    try:
        torso_vector = hip_mid - shoulder_mid
        if np.linalg.norm(torso_vector) < 10:
            return 90.0
        
        vertical_vector = np.array([0, 1])
        dot_product = np.dot(torso_vector, vertical_vector)
        norms = np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector)
        cosine_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    except:
        return 90.0

def calculate_angle(a, b, c):
    """Calculate angle at point b."""
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        
        if np.linalg.norm(ba) < 1 or np.linalg.norm(bc) < 1:
            return 180.0
        
        dot_product = np.dot(ba, bc)
        norms = np.linalg.norm(ba) * np.linalg.norm(bc)
        cosine_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    except:
        return 180.0

def get_person_bbox(keypoints):
    """Calculate bounding box from keypoints."""
    try:
        valid_points = []
        for kp in keypoints:
            if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                valid_points.append(kp[:2])
        
        if len(valid_points) < 3:
            return None
        
        valid_points = np.array(valid_points)
        x_min, y_min = valid_points.min(axis=0)
        x_max, y_max = valid_points.max(axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, int(x_min) - padding)
        y_min = max(0, int(y_min) - padding)
        x_max = int(x_max) + padding
        y_max = int(y_max) + padding
        
        return [x_min, y_min, x_max, y_max]
    except:
        return None

def init_csv():
    """Initialize CSV file."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'datetime', 'pose', 'person_id', 
                'confidence', 'torso_angle', 'knee_angle', 'hip_angle',
                'long_term_alert', 'alert_duration', 'alert_count'
            ])

def log_pose_to_csv(timestamp, datetime_str, pose, person_id, confidence, 
                   torso_angle, knee_angle, hip_angle, long_term_alert=False, 
                   alert_duration=0, alert_count=0):
    """Log pose data to CSV."""
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, datetime_str, pose, person_id,
                f"{confidence:.3f}", f"{torso_angle:.1f}", 
                f"{knee_angle:.1f}", f"{hip_angle:.1f}",
                long_term_alert, f"{alert_duration:.1f}", alert_count
            ])
    except Exception as e:
        print(f"⚠️ CSV write error: {str(e)[:50]}")

def save_alert_image_ultra(frame, person_id, duration=0, alert_count=0):
    """Save alert image with complete skeleton visualization."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{alert_count}_person{person_id}_{timestamp}.jpg"
        image_path = os.path.join(ALERT_IMAGES_DIR, filename)
        
        annotated_frame = frame.copy()
        
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (5, 5), (450, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        
        cv2.putText(annotated_frame, f"ALERT #{alert_count}", (15, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Person ID: {person_id} | Duration: {duration:.1f}s", (15, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imwrite(image_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"💾 Alert image saved: {filename}")
        return image_path
        
    except Exception as e:
        print(f"⚠️ Save image error: {str(e)[:50]}")
        return ""

def draw_person_box_with_pose(frame, tracker):
    """Draw bounding box with pose label for each person."""
    bbox = tracker.bbox
    pose = tracker.pose
    confidence = tracker.confidence
    person_id = tracker.id
    
    # Box color based on pose
    if pose == 'lying_down':
        box_color = (0, 0, 255)  # Red
        text_bg_color = (0, 0, 200)
    elif pose == 'sitting':
        box_color = (255, 255, 0)  # Cyan
        text_bg_color = (200, 200, 0)
    else:
        box_color = (0, 255, 0)  # Green
        text_bg_color = (0, 200, 0)
    
    # Draw bounding box
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 3)
    
    # Prepare label text
    label = f"ID:{person_id} {pose.upper()}"
    conf_text = f"{confidence:.2f}"
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    thickness = 2
    
    (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, 0.5, 1)
    
    # Draw label background (above box)
    label_y = bbox[1] - 10
    if label_y < label_h + 10:
        label_y = bbox[1] + label_h + 10
    
    cv2.rectangle(frame, 
                 (bbox[0], label_y - label_h - 10),
                 (bbox[0] + max(label_w, conf_w) + 20, label_y + 5),
                 text_bg_color, -1)
    
    # Draw text
    cv2.putText(frame, label, (bbox[0] + 10, label_y - 5), 
               font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, conf_text, (bbox[0] + 10, label_y + conf_h + 10), 
               font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # If lying down, add warning icon
    if pose == 'lying_down':
        cv2.putText(frame, "⚠️", (bbox[2] - 40, bbox[1] + 40), 
                   font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

# =========================
# MAIN VIDEO PROCESSING
# =========================
def process_video_multi_person(video_path):
    """
    Multi-person pose detection with:
    - Individual tracking for each person
    - Bounding boxes with pose labels
    - Alert if ANY person falls down
    """
    init_csv()
    human_validator = HumanValidator()
    person_tracker = MultiPersonTracker()
    alert_system = MultiPersonAlertSystem(initial_threshold=10, max_alerts=3, cooldown=10)
    
    csv_monitor = UltraFastCSVMonitor(CSV_FILE, check_interval=2)
    csv_monitor.start_monitoring()
    
    print(f"\n{'='*60}")
    print(f"🚀 MULTI-PERSON POSE DETECTION SYSTEM")
    print(f"{'='*60}")
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"👥 Multi-person tracking: ENABLED")
    print(f"⚡ Alert: 10s initial, 10s cooldown, max 3 per person")
    print(f"🎯 Image: {ALERT_IMAGE_WIDTH}x{ALERT_IMAGE_HEIGHT}, quality={ALERT_IMAGE_QUALITY}")
    print(f"{'='*60}\n")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        csv_monitor.stop_monitoring()
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video Info:")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Total Frames: {total_frames}")
    print(f"   - Resolution: {width}x{height}")
    print(f"\n{'='*60}\n")
    
    cv2.namedWindow("Multi-Person Pose Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-Person Pose Detection", 1200, 800)
    
    last_log_time = defaultdict(float)
    log_interval = 2.0
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps_counter = 0
    display_fps = 0
    
    paused = False
    
    print("▶️  Processing started...")
    print("📋 Controls: 'q'=quit, 'p'=pause/resume")
    print(f"\n{'='*60}\n")
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✅ End of video reached")
                    break

                frame_count += 1
                current_time = time.time()
                
                fps_counter += 1
                if current_time - last_fps_time >= 1.0:
                    display_fps = fps_counter
                    fps_counter = 0
                    last_fps_time = current_time
                
                # YOLO inference
                results = pose_model(
                    frame, 
                    verbose=False, 
                    conf=YOLO_CONF_THRESHOLD,
                    imgsz=YOLO_IMG_SIZE,
                    half=False
                )
                
                display_frame = frame.copy()
                detections = []
                
                # Process ALL detections (multi-person)
                if (len(results) > 0 and 
                    results[0].keypoints is not None and 
                    len(results[0].keypoints.xy) > 0):
                    
                    all_keypoints = results[0].keypoints.xy.cpu().numpy()
                    
                    # Process each detected person
                    for person_idx, keypoints in enumerate(all_keypoints):
                        if human_validator.validate_human_presence(keypoints):
                            # Extract features
                            features = extract_pose_features(keypoints)
                            
                            if features is not None:
                                # Classify pose
                                pose_label, confidence_scores, _ = classify_pose_accurate(
                                    features, "standing"
                                )
                                
                                main_confidence = max(confidence_scores.values())
                                
                                # Get bounding box
                                bbox = get_person_bbox(keypoints)
                                
                                if bbox is not None and pose_label != "uncertain":
                                    detections.append((bbox, keypoints, pose_label, 
                                                     main_confidence, features))
                
                # Update trackers with detections
                active_ids = person_tracker.update(detections, current_time)
                active_trackers = person_tracker.get_active_trackers(current_time)
                
                # Process each active person
                for tracker_id, tracker in active_trackers.items():
                    # Draw skeleton
                    display_frame = draw_complete_skeleton(
                        display_frame, tracker.keypoints, tracker.pose
                    )
                    
                    # Draw bounding box with label
                    draw_person_box_with_pose(display_frame, tracker)
                    
                    # Log to CSV periodically
                    if (current_time - last_log_time[tracker_id] >= log_interval and 
                        tracker.pose != "uncertain" and tracker.features is not None):
                        
                        log_pose_to_csv(
                            timestamp=current_time,
                            datetime_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            pose=tracker.pose,
                            person_id=tracker_id,
                            confidence=tracker.confidence,
                            torso_angle=tracker.features['torso_angle'],
                            knee_angle=tracker.features['avg_knee_angle'],
                            hip_angle=tracker.features['avg_hip_angle']
                        )
                        
                        last_log_time[tracker_id] = current_time
                    
                    # Check for alerts
                    if tracker.pose == 'lying_down' and tracker.confidence > 0.55:
                        alert_triggered, lying_duration, alert_count, person_id = \
                            alert_system.update_person(tracker, current_time)
                        
                        if alert_triggered:
                            # Save alert image
                            alert_image_path = save_alert_image_ultra(
                                display_frame, person_id, lying_duration, alert_count
                            )
                            
                            print(f"\n{'='*60}")
                            print(f"🚨 ALERT #{alert_count} - PERSON {person_id}")
                            print(f"⏱️  Duration: {lying_duration:.1f}s")
                            print(f"🎯 Confidence: {tracker.confidence:.2f}")
                            print(f"{'='*60}\n")
                            
                            # Log alert to CSV
                            log_pose_to_csv(
                                timestamp=current_time,
                                datetime_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                pose='lying_down',
                                person_id=person_id,
                                confidence=tracker.confidence,
                                torso_angle=tracker.features['torso_angle'],
                                knee_angle=tracker.features['avg_knee_angle'],
                                hip_angle=tracker.features['avg_hip_angle'],
                                long_term_alert=True,
                                alert_duration=lying_duration,
                                alert_count=alert_count
                            )
                            
                            # Send alert
                            alert_message = f"""🚨 POSE ALERT #{alert_count}

👤 Person ID: {person_id}
⏰ Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
⏱️ Duration: {lying_duration:.1f}s
🎯 Confidence: {tracker.confidence:.2f}
📍 Status: Person lying down

⚠️ Please check immediately!"""
                            
                            send_alert_ultra_fast(alert_message, alert_image_path)
            
            # Display info overlay
            info_overlay = display_frame.copy()
            
            panel_height = 120
            cv2.rectangle(info_overlay, (0, 0), (600, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(info_overlay, 0.6, display_frame, 0.4, 0, display_frame)
            
            y_pos = 30
            line_height = 30
            
            # Multi-person stats
            active_trackers = person_tracker.get_active_trackers(current_time)
            num_people = len(active_trackers)
            
            cv2.putText(display_frame, f"PEOPLE DETECTED: {num_people}", 
                       (15, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            y_pos += line_height
            
            if num_people > 0:
                poses_summary = {}
                for tracker in active_trackers.values():
                    poses_summary[tracker.pose] = poses_summary.get(tracker.pose, 0) + 1
                
                summary_text = " | ".join([f"{pose}: {count}" for pose, count in poses_summary.items()])
                cv2.putText(display_frame, summary_text, 
                           (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                y_pos += line_height
            
            # Total alerts
            cv2.putText(display_frame, f"Total Alerts: {alert_system.global_alert_count}", 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Performance info
            perf_text = f"FPS: {display_fps} | Frame: {frame_count}/{total_frames}"
            text_size = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = display_frame.shape[1] - text_size[0] - 15
            text_y = display_frame.shape[0] - 15
            
            cv2.rectangle(display_frame, 
                         (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5),
                         (0, 0, 0), -1)
            cv2.putText(display_frame, perf_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Pause indicator
            if paused:
                pause_text = "PAUSED (Press 'p' to resume)"
                text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
                text_x = (display_frame.shape[1] - text_size[0]) // 2
                text_y = (display_frame.shape[0] + text_size[1]) // 2
                
                cv2.rectangle(display_frame,
                             (text_x - 20, text_y - text_size[1] - 20),
                             (text_x + text_size[0] + 20, text_y + 20),
                             (0, 0, 0), -1)
                cv2.putText(display_frame, pause_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Multi-Person Pose Detection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n⚠️  User requested quit")
                break
            elif key == ord("p"):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"{'⏸️' if paused else '▶️'}  {status}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n{'='*60}")
        print("🧹 Cleaning up...")
        
        cap.release()
        cv2.destroyAllWindows()
        csv_monitor.stop_monitoring()
        
        processing_time = time.time() - start_time
        avg_fps = frame_count / processing_time if processing_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✅ PROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"📊 Statistics:")
        print(f"   - Total Frames: {frame_count}")
        print(f"   - Processing Time: {processing_time:.1f}s")
        print(f"   - Average FPS: {avg_fps:.1f}")
        print(f"   - Total Alerts: {alert_system.global_alert_count}")
        print(f"   - Max People Detected: {person_tracker.next_id - 1}")
        print(f"   - CSV Log: {CSV_FILE}")
        print(f"   - Alert Images: {ALERT_IMAGES_DIR}")
        print(f"{'='*60}\n")

# =========================
# MAIN APPLICATION
# =========================
def main():
    """Main application entry point."""
    print(f"\n{'='*60}")
    print("🚀 MULTI-PERSON POSE DETECTION SYSTEM")
    print(f"{'='*60}\n")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        print(f"Please update VIDEO_PATH in the script")
        return
    
    print(f"✅ Video file found: {os.path.basename(VIDEO_PATH)}")
    print(f"\n{'='*60}\n")
    
    print("🧪 Testing notification system...\n")
    
    test_message = """⚡ MULTI-PERSON POSE DETECTION SYSTEM

✅ System Started Successfully
👥 Multi-person tracking enabled
📹 Video processing initialized
🎯 Alert system ready
📊 All systems operational

This is a test notification."""
    
    try:
        send_alert_ultra_fast(test_message)
        print("\n✅ Notification test completed")
    except Exception as e:
        print(f"\n⚠️  Notification test warning: {str(e)[:50]}")
        print("   (Processing will continue)")
    
    print(f"\n{'='*60}\n")
    input("Press ENTER to start video processing...")
    print()
    
    try:
        process_video_multi_person(VIDEO_PATH)
    except Exception as e:
        print(f"\nâŒ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Program ended\n")

if __name__ == "__main__":
    main()
    