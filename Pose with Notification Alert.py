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

# =========================
# PERFORMANCE CONFIGURATION
# =========================
ALERT_IMAGE_QUALITY = 70
MAX_IMAGE_SIZE = (640, 480)
TELEGRAM_TIMEOUT = 5
EMAIL_TIMEOUT = 10

# =========================
# NOTIFICATION CONFIGURATION
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
    "hrithiksai.n2021@vitstudent.ac.in","sweta.b@vit.ac.in","senthilkumar.t@vit.ac.in"
]

# =========================
# ASYNC EVENT LOOP SETUP
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
                except RuntimeError:
                    cls._instance = asyncio.new_event_loop()
                    asyncio.set_event_loop(cls._instance)
            return cls._instance

# =========================
# HIGH-PERFORMANCE NOTIFICATION FUNCTIONS
# =========================
async def send_telegram_message_quick(session, message: str):
    """Send text message with timeout."""
    try:
        url = f"{BASE_URL}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        async with session.post(url, data=payload, timeout=aiohttp.ClientTimeout(total=TELEGRAM_TIMEOUT)) as resp:
            return await resp.json()
    except asyncio.TimeoutError:
        print("❌ Telegram message timeout")
        return None
    except Exception as e:
        print(f"❌ Telegram message error: {e}")
        return None

async def send_telegram_photo_quick(session, image_path: str, caption: str = "📷 Alert Evidence"):
    """Send optimized photo with timeout."""
    try:
        # Optimize image before sending
        optimized_path = await optimize_image_for_alert(image_path)
        
        url = f"{BASE_URL}/sendPhoto"
        with open(optimized_path, "rb") as photo:
            form = aiohttp.FormData()
            form.add_field("chat_id", CHAT_ID)
            form.add_field("caption", caption)
            form.add_field("photo", photo, filename=os.path.basename(optimized_path), content_type="image/jpeg")
            async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=TELEGRAM_TIMEOUT)) as resp:
                # Clean up optimized image
                if optimized_path != image_path:
                    try:
                        os.remove(optimized_path)
                    except:
                        pass
                return await resp.json()
    except asyncio.TimeoutError:
        print("❌ Telegram photo timeout")
        return None
    except Exception as e:
        print(f"❌ Telegram photo error: {e}")
        return None

async def optimize_image_for_alert(image_path: str) -> str:
    """Optimize image for faster transmission."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
            
        # Resize image
        height, width = img.shape[:2]
        if height > MAX_IMAGE_SIZE[1] or width > MAX_IMAGE_SIZE[0]:
            img = cv2.resize(img, MAX_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        
        # Create optimized version
        optimized_path = image_path.replace('.jpg', '_optimized.jpg')
        cv2.imwrite(optimized_path, img, [cv2.IMWRITE_JPEG_QUALITY, ALERT_IMAGE_QUALITY])
        return optimized_path
    except Exception as e:
        print(f"❌ Image optimization error: {e}")
        return image_path

def send_email_quick(subject: str, body: str, image_path: str = None):
    """Send email with timeout and optimization."""
    for sender in SENDER_ACCOUNTS:
        try:
            msg = MIMEMultipart()
            msg['From'] = sender["email"]
            msg['To'] = ", ".join(RECEIVER_EMAILS)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, "plain"))

            # Optimize and attach image if provided
            if image_path and os.path.exists(image_path):
                # Run optimization in current thread
                loop = AsyncEventLoop.get_loop()
                optimized_path = loop.run_until_complete(optimize_image_for_alert(image_path))
                
                with open(optimized_path, "rb") as file:
                    mime = MIMEBase("application", "octet-stream")
                    mime.set_payload(file.read())
                    encoders.encode_base64(mime)
                    mime.add_header("Content-Disposition", f"attachment; filename={os.path.basename(optimized_path)}")
                    msg.attach(mime)
                
                # Clean up optimized image
                if optimized_path != image_path:
                    try:
                        os.remove(optimized_path)
                    except:
                        pass

            # Send with timeout
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=EMAIL_TIMEOUT)
            server.starttls()  
            server.login(sender["email"], sender["password"])
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email sent quickly from {sender['email']}")
            return True
            
        except Exception as e:
            print(f"❌ Quick email failed from {sender['email']}: {e}")
            continue
    
    return False

async def send_telegram_alert_quick(message: str, image_path: str = None):
    """Send Telegram alert quickly."""
    async with aiohttp.ClientSession() as session:
        tasks = [send_telegram_message_quick(session, message)]
        if image_path and os.path.exists(image_path):
            tasks.append(send_telegram_photo_quick(session, image_path, message))
        
        # Don't wait too long for results
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=TELEGRAM_TIMEOUT * 2)
        except asyncio.TimeoutError:
            print("⚠️ Telegram alert timed out, continuing...")
        
    print("✅ Telegram notification sent quickly")

def send_quick_alert_sync(message: str, image_path: str = None):
    """Synchronous wrapper for quick alerts."""
    loop = AsyncEventLoop.get_loop()
    
    # Run Telegram async
    telegram_task = asyncio.run_coroutine_threadsafe(
        send_telegram_alert_quick(message, image_path), 
        loop
    )
    
    # Run email sync
    email_success = send_email_quick("🚨 Pose Alert", message, image_path)
    
    # Wait for telegram with timeout
    try:
        telegram_task.result(timeout=TELEGRAM_TIMEOUT * 2)
    except:
        print("⚠️ Telegram task timed out")
    
    return email_success

# =========================
# SIMPLIFIED CSV MONITORING
# =========================
class FastCSVMonitor:
    def __init__(self, csv_file_path, check_interval=3):
        self.csv_file = csv_file_path
        self.check_interval = check_interval
        self.last_processed_index = 0
        self.sent_alerts = set()
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start monitoring without complex async."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"🔍 FAST CSV monitoring started: {self.csv_file}")
        
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("⏹️ Stopped CSV monitoring")
        
    def _monitor_loop(self):
        """Simple monitoring loop."""
        while self.monitoring:
            try:
                self._check_new_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"❌ CSV monitoring error: {e}")
                time.sleep(5)
                
    def _check_new_alerts(self):
        """Check for new alerts and process immediately."""
        if not os.path.exists(self.csv_file):
            return
            
        try:
            # Read CSV efficiently
            df = pd.read_csv(self.csv_file)
            
            if len(df) == 0:
                return
                
            # Find new alerts
            new_alerts = df[
                (df['long_term_alert'] == True) & 
                (df.index >= self.last_processed_index)
            ]
            
            for index, row in new_alerts.iterrows():
                alert_id = f"{row['timestamp']}_{row['alert_count']}"
                if alert_id not in self.sent_alerts:
                    self._process_alert_immediately(row, alert_id)
                    self.sent_alerts.add(alert_id)
                    
            if len(df) > 0:
                self.last_processed_index = len(df) - 1
                
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            
    def _process_alert_immediately(self, row, alert_id):
        """Process alert immediately without async complexity."""
        try:
            timestamp = row['datetime']
            duration = row['alert_duration']
            alert_count = row['alert_count']
            confidence = row['confidence']
            
            message = f"""🚨 FAST ALERT #{alert_count}

⏰ Time: {timestamp}
⏱️ Duration: {duration:.1f}s
🎯 Confidence: {confidence:.2f}

Quick alert - person lying down."""

            image_path = self._find_alert_image(timestamp, alert_count)
            
            # Send alert immediately
            send_quick_alert_sync(message, image_path)
            
            print(f"⚡ Quick alert sent: {alert_id}")
            
        except Exception as e:
            print(f"❌ Alert processing error: {e}")

    def _find_alert_image(self, timestamp, alert_count):
        """Quick image finder."""
        try:
            alerts_dir = os.path.join(os.path.dirname(self.csv_file), "alerts")
            if not os.path.exists(alerts_dir):
                return None
                
            # Quick search for latest matching file
            files = os.listdir(alerts_dir)
            for filename in files:
                if f"alert{alert_count}" in filename:
                    return os.path.join(alerts_dir, filename)
                    
            # Fallback: look for any alert image
            for filename in files:
                if filename.endswith('.jpg') and 'alert' in filename.lower():
                    return os.path.join(alerts_dir, filename)
                    
            return None
        except:
            return None

# =========================
# OPTIMIZED POSE DETECTION
# =========================
OUTPUT_DIR = "pose_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(OUTPUT_DIR, "pose_analysis.csv")
ALERT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "alerts")
os.makedirs(ALERT_IMAGES_DIR, exist_ok=True)

VIDEO_PATH = r"D:\Perception Track.ai\dataset\VID_20251007_180327395~2.mp4"

# Load YOLO model
try:
    pose_model = YOLO("yolov8s-pose.pt")
    print("✅ YOLO Pose model loaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    exit(1)

# =========================
# POSE CLASSIFICATION
# =========================
def classify_pose_improved(features, previous_pose="standing"):
    torso = features['torso_angle']
    knee = features['avg_knee_angle']
    hip = features['avg_hip_angle']
    
    scores = {'standing': 0.0, 'sitting': 0.0, 'lying_down': 0.0}
    
    if torso >= 55:
        lying_score = 0.7
        if torso >= 70: lying_score = 0.85
        if torso >= 85: lying_score = 0.95
        scores['lying_down'] = lying_score
    
    elif knee < 145:
        sitting_score = 0.6
        if 60 <= knee <= 135: sitting_score += 0.2
        if 80 <= hip <= 140: sitting_score += 0.15
        scores['sitting'] = min(sitting_score, 1.0)
    
    elif torso <= 40 and knee >= 150:
        standing_score = 0.7
        if torso <= 20: standing_score += 0.2
        if knee >= 165: standing_score += 0.1
        scores['standing'] = min(standing_score, 1.0)
    
    else:
        if previous_pose in scores:
            scores[previous_pose] = 0.4
    
    best_pose = max(scores, key=scores.get)
    best_confidence = scores[best_pose]
    
    if best_confidence < 0.5:
        return "uncertain", scores, previous_pose
    
    return best_pose, scores, best_pose

# =========================
# FAST ALERT SYSTEM
# =========================
class FastAlertSystem:
    def __init__(self, initial_threshold=10, max_alerts=3):
        self.initial_threshold = initial_threshold
        self.max_alerts = max_alerts
        self.lying_start_time = None
        self.consecutive_lying_frames = 0
        self.min_consecutive_frames = 3
        self.alert_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 15
        self.alert_triggered = False
        
    def update(self, current_pose, current_time):
        if current_pose == 'lying_down':
            self.consecutive_lying_frames += 1
            
            if self.consecutive_lying_frames >= self.min_consecutive_frames:
                if self.lying_start_time is None:
                    self.lying_start_time = current_time
                    print(f"⚡ Lying detected - fast timer started")
                
                lying_duration = current_time - self.lying_start_time
                
                # First alert
                if (lying_duration >= self.initial_threshold and 
                    not self.alert_triggered and 
                    self.alert_count < self.max_alerts):
                    
                    self.alert_triggered = True
                    self.alert_count += 1
                    self.last_alert_time = current_time
                    return True, lying_duration, self.alert_count
                
                # Subsequent alerts
                elif (self.alert_triggered and 
                      current_time - self.last_alert_time >= self.alert_cooldown and
                      self.alert_count < self.max_alerts):
                    
                    self.alert_count += 1
                    self.last_alert_time = current_time
                    return True, lying_duration, self.alert_count
                    
            return False, 0, self.alert_count
            
        else:
            if self.consecutive_lying_frames > 0:
                self.consecutive_lying_frames = max(0, self.consecutive_lying_frames - 2)
                if self.consecutive_lying_frames == 0:
                    self.lying_start_time = None
                    self.alert_triggered = False
            return False, 0, self.alert_count

# =========================
# UTILITY FUNCTIONS
# =========================
class HumanValidator:
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
    try:
        keypoints_list = []
        for i in [5, 6, 11, 12, 13, 14, 15, 16]:
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

def init_csv():
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
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, datetime_str, pose, person_id,
            f"{confidence:.3f}", f"{torso_angle:.1f}", 
            f"{knee_angle:.1f}", f"{hip_angle:.1f}",
            long_term_alert, f"{alert_duration:.1f}", alert_count
        ])

def save_alert_image_fast(frame, alert_type, person_id, duration=0, alert_count=0):
    """Save optimized alert image quickly."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if alert_type == "long_term_lying":
        filename = f"alert_{alert_count}_{timestamp}.jpg"
        image_path = os.path.join(ALERT_IMAGES_DIR, filename)
        
        # Simple annotation for speed
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame, f"ALERT #{alert_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"{duration:.1f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save with lower quality
        cv2.imwrite(image_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, ALERT_IMAGE_QUALITY])
        return image_path
    
    return ""

def draw_skeleton_fast(frame, keypoints, pose_label):
    """Fast skeleton drawing."""
    if keypoints is None or len(keypoints) < 17:
        return frame
    
    # Simple skeleton connections
    skeleton = [(5, 6), (5, 11), (6, 12), (11, 12)]
    
    if pose_label == 'lying_down': color = (0, 255, 255)
    elif pose_label == 'sitting': color = (255, 255, 0)
    else: color = (0, 255, 0)
    
    for start, end in skeleton:
        if (start < len(keypoints) and end < len(keypoints) and
            len(keypoints[start]) >= 2 and len(keypoints[end]) >= 2 and
            keypoints[start][0] > 0 and keypoints[start][1] > 0 and
            keypoints[end][0] > 0 and keypoints[end][1] > 0):
            
            start_point = (int(keypoints[start][0]), int(keypoints[start][1]))
            end_point = (int(keypoints[end][0]), int(keypoints[end][1]))
            cv2.line(frame, start_point, end_point, color, 2)
    
    return frame

# =========================
# FAST MAIN PROCESSING
# =========================
def process_video_fast(video_path):
    # Initialize fast systems
    init_csv()
    human_validator = HumanValidator()
    alert_system = FastAlertSystem(initial_threshold=10, max_alerts=3)
    
    # Start FAST CSV monitoring
    csv_monitor = FastCSVMonitor(CSV_FILE, check_interval=3)
    csv_monitor.start_monitoring()
    
    print(f"⚡ FAST Processing: {video_path}")
    print(f"🚀 Alert System: 10s initial, 15s cooldown, max 3 alerts")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        csv_monitor.stop_monitoring()
        return
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video: {fps:.2f} FPS, {total_frames} frames")
    
    cv2.namedWindow("Fast Pose Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fast Pose Analysis", 1000, 700)
    
    previous_pose = "standing"
    person_id = 1
    last_log_time = 0
    log_interval = 2.0
    
    frame_count = 0
    start_time = time.time()
    
    print("[INFO] FAST pose classification started...")
    print("[INFO] Controls: 'q'=quit, 'p'=pause")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("✅ End of video reached")
                    break

                frame_count += 1
                current_time = time.time()
                
                # Fast YOLO inference
                results = pose_model(frame, verbose=False, conf=0.6)
                
                human_detected = False
                current_pose = "no_person"
                display_frame = frame.copy()
                
                if (len(results) > 0 and results[0].keypoints is not None and 
                    len(results[0].keypoints.xy) > 0):
                    
                    keypoints = results[0].keypoints.xy.cpu().numpy()[0]
                    
                    if human_validator.validate_human_presence(keypoints):
                        human_detected = True
                        
                        features = extract_pose_features(keypoints)
                        
                        if features is not None:
                            pose_label, confidence_scores, previous_pose = classify_pose_improved(
                                features, previous_pose
                            )
                            
                            main_confidence = max(confidence_scores.values())
                            current_pose = pose_label
                            
                            # Draw simple skeleton
                            display_frame = draw_skeleton_fast(display_frame, keypoints, pose_label)
                            
                            # Less frequent CSV logging
                            if (current_time - last_log_time >= log_interval and 
                                pose_label != "uncertain"):
                                
                                timestamp = current_time
                                datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                log_pose_to_csv(
                                    timestamp=timestamp,
                                    datetime_str=datetime_str,
                                    pose=pose_label,
                                    person_id=person_id,
                                    confidence=main_confidence,
                                    torso_angle=features['torso_angle'],
                                    knee_angle=features['avg_knee_angle'],
                                    hip_angle=features['avg_hip_angle']
                                )
                                
                                last_log_time = current_time
                            
                            # FAST ALERT SYSTEM
                            if pose_label == 'lying_down' and main_confidence > 0.6:
                                alert_triggered, lying_duration, alert_count = alert_system.update(
                                    pose_label, current_time
                                )
                                
                                if alert_triggered:
                                    alert_image_path = save_alert_image_fast(
                                        frame, "long_term_lying", person_id, lying_duration, alert_count
                                    )
                                    print(f"⚡ ALERT #{alert_count}! Lying for {lying_duration:.1f}s")
                                    
                                    # Fast CSV logging
                                    log_pose_to_csv(
                                        timestamp=current_time,
                                        datetime_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        pose='lying_down',
                                        person_id=person_id,
                                        confidence=main_confidence,
                                        torso_angle=features['torso_angle'],
                                        knee_angle=features['avg_knee_angle'],
                                        hip_angle=features['avg_hip_angle'],
                                        long_term_alert=True,
                                        alert_duration=lying_duration,
                                        alert_count=alert_count
                                    )
            
            # Fast display
            if human_detected and features is not None:
                color = (0, 255, 0)
                if pose_label == 'lying_down': color = (0, 255, 255)
                elif pose_label == 'sitting': color = (255, 255, 0)
                
                cv2.putText(display_frame, f"POSE: {pose_label.upper()}", (15, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                if alert_system.alert_count > 0:
                    cv2.putText(display_frame, f"Alerts: {alert_system.alert_count}", 
                               (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Fast Pose Analysis", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")

    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted")
    except Exception as e:
        print(f"❌ Processing error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_monitor.stop_monitoring()
        
        processing_time = time.time() - start_time
        print(f"\n🎉 FAST PROCESSING COMPLETED!")
        print(f"⚡ Total alerts: {alert_system.alert_count}")
        print(f"⏱️ Processing time: {processing_time:.1f}s")

# =========================
# Fast Main Application
# =========================
def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        return
    
    print("🚀 Testing FAST notification system...")
    
    # Test notifications
    test_message = "⚡ FAST Pose Analysis System Started!\n\nOptimized for speed with quick alerts."
    send_quick_alert_sync(test_message)
    
    # Start fast processing
    process_video_fast(VIDEO_PATH)

if __name__ == "__main__":
    main()