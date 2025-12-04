import os
import time
import cv2
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
import csv
from collections import defaultdict
import re
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

warnings.filterwarnings("ignore")


VIDEO_PATH = r"D:\Perception Track.ai\dataset\1.mp4"
SAVE_DIR = r"D:\Perception Track.ai\unified_output 2"


YOLO_OBJECT_MODEL = "yolov8n Hrithik Vehicles.pt" 
YOLO_POSE_MODEL = "yolov8n-pose.pt" 
PLATE_YOLO_MODEL = "license_plate_detector.pt"
USE_PLATE_YOLO = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


OBJECT_DETECTION_CONFIDENCE = 0.35
POSE_DETECTION_CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 15
YOLO_IMG_SIZE = 640


MIN_PERSON_QUALITY = 30
MIN_PERSON_AREA = 8000
MIN_PERSON_WIDTH = 80
MIN_PERSON_HEIGHT = 150
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 2.0
PERSON_TRACKING_FRAMES = 8


MIN_VEHICLE_QUALITY = 25
MIN_VEHICLE_AREA = 8000


VEHICLE_SIMILARITY_THRESHOLD = 0.92
PERSON_SIMILARITY_THRESHOLD = 0.88
MIN_TIME_BETWEEN_SAME_VEHICLE = 10


ENABLE_FACE_RECOGNITION = True
FACE_RECOGNITION_ASYNC = True
FACE_CHECK_INTERVAL = 5
FACE_ENCODINGS_FILE = "face_encodings_pca.pkl"
FACE_TOLERANCE = 0.5
MAX_FACE_WORKERS = 1
MAX_FACE_ATTEMPTS = 3


ENABLE_PLATE_OCR = True
OCR_BACKEND = "easyocr"
MAX_OCR_WORKERS = 1
PLATE_MIN_CONFIDENCE = 0.3

POSE_ALERT_ENABLED = True
LYING_ALERT_THRESHOLD = 10 
MAX_ALERTS_PER_PERSON = 3
ALERT_COOLDOWN = 10  
MIN_CONSECUTIVE_FRAMES = 5

TELEGRAM_ENABLED = True
EMAIL_ENABLED = True
TELEGRAM_TIMEOUT = 3
EMAIL_TIMEOUT = 8

BOT_TOKEN = "8331904126:AAGjeeo7hWO-CHAsYAhQ7ZctRSNqpUwBAo4"
CHAT_ID = "1490935977"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

SENDER_ACCOUNTS = [
    {"email": "boostup1947@gmail.com",
      "password": "bkcl hcuv gcjf gslz"}
]
RECEIVER_EMAILS = [

    "hrithiksai.n2021@vitstudent.ac.in"
]

ALERT_IMAGE_WIDTH = 640
ALERT_IMAGE_HEIGHT = 480
ALERT_IMAGE_QUALITY = 65
MAX_IMAGE_SIZE_BYTES = 500 * 1024

SHOW_VIDEO = True
DISPLAY_WIDTH = 1280

GPU_MEMORY_FRACTION = 0.55
CUDA_BENCHMARK = True


os.makedirs(SAVE_DIR, exist_ok=True)
for subdir in ["vehicles", "persons", "plates", "faces", "logs", "pose_alerts"]:
    os.makedirs(os.path.join(SAVE_DIR, subdir), exist_ok=True)


VEHICLE_CSV = os.path.join(SAVE_DIR, "logs", "vehicles.csv")
PERSON_CSV = os.path.join(SAVE_DIR, "logs", "persons.csv")
FACE_CSV = os.path.join(SAVE_DIR, "logs", "face_matches.csv")
PLATE_CSV = os.path.join(SAVE_DIR, "logs", "plates.csv")
POSE_CSV = os.path.join(SAVE_DIR, "logs", "pose_analysis.csv")

def init_csv(path, headers):
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

init_csv(VEHICLE_CSV, ["timestamp", "track_id", "vehicle_type", "color", "confidence", 
                       "plate_number", "quality", "image_path", "first_seen", "last_seen"])
init_csv(PERSON_CSV, ["timestamp", "track_id", "name", "confidence", "quality", 
                      "image_path", "first_seen", "last_seen", "face_matches"])
init_csv(FACE_CSV, ["timestamp", "track_id", "person_name", "confidence", "quality"])
init_csv(PLATE_CSV, ["timestamp", "track_id", "plate_text", "confidence", "vehicle_type", "method"])
init_csv(POSE_CSV, ['timestamp', 'datetime', 'pose', 'person_id', 'confidence', 
                    'torso_angle', 'knee_angle', 'hip_angle', 'long_term_alert', 
                    'alert_duration', 'alert_count'])

csv_lock = threading.Lock()

def append_csv(path, row):
    with csv_lock:
        try:
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([str(x).replace(',', ';') for x in row])
        except Exception as e:
            print(f"⚠️ CSV error: {e}")

# gpu hrithik opti#
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    torch.backends.cudnn.benchmark = CUDA_BENCHMARK
    torch.cuda.empty_cache()
    print(f"✅ CUDA initialized: {torch.cuda.get_device_name(0)}")

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

# IMAGE ALERTS
def optimize_image_ultra_fast(image_path: str) -> str:
    """Aggressive image optimization for fastest transmission."""
    try:
        img = cv2.imread(image_path)
        if img is None:
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
                    return optimized_path
            quality -= 10
        
        cv2.imwrite(optimized_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return optimized_path
        
    except Exception as e:
        print(f"⚠️ Image optimization error: {e}")
        return image_path

# TELEGRAM noti

async def send_telegram_message_ultra(session, message: str):
    """Send Telegram message with minimal timeout."""
    if not TELEGRAM_ENABLED:
        return None
        
    try:
        url = f"{BASE_URL}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        
        timeout = aiohttp.ClientTimeout(total=TELEGRAM_TIMEOUT)
        async with session.post(url, data=payload, timeout=timeout) as resp:
            if resp.status == 200:
                print("✅ Telegram message sent")
                return await resp.json()
            else:
                print(f"⚠️ Telegram status: {resp.status}")
                return None
                
    except asyncio.TimeoutError:
        print("⚠️ Telegram timeout")
        return None
    except Exception as e:
        print(f"⚠️ Telegram error: {str(e)[:50]}")
        return None

async def send_telegram_photo_ultra(session, image_path: str, caption: str):
    """Send optimized photo ultra-fast."""
    if not TELEGRAM_ENABLED:
        return None
        
    optimized_path = None
    try:
        optimized_path = optimize_image_ultra_fast(image_path)
        
        if not os.path.exists(optimized_path):
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
                    return None
                    
    except asyncio.TimeoutError:
        print("⚠️ Telegram photo timeout")
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
        print("⚠️ Telegram alert timeout")
    except Exception as e:
        print(f"⚠️ Telegram alert error: {str(e)[:50]}")

# EMAIL NOTI

def send_email_ultra(subject: str, body: str, image_path: str = None):
    """Send email ultra-fast with optimization."""
    if not EMAIL_ENABLED:
        return False
        
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
            
        except Exception as e:
            print(f"⚠️ Email error: {str(e)[:50]}")
            continue
        finally:
            if optimized_path and optimized_path != image_path:
                try:
                    if os.path.exists(optimized_path):
                        os.remove(optimized_path)
                except:
                    pass
    
    return False

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
            send_email_ultra("🚨 Detection Alert", message, image_path)
        except Exception as e:
            print(f"⚠️ Email task error: {str(e)[:50]}")
    
    executor = ThreadPoolExecutor(max_workers=2)
    executor.submit(send_telegram_task)
    executor.submit(send_email_task)
    
    print("⚡ Alerts dispatched in parallel")

# COLOR DETECTION
COLOR_RANGES = {
    'dark_red': ([0, 100, 50], [10, 255, 150]),
    'red': ([0, 120, 70], [10, 255, 255]),
    'bright_red': ([170, 120, 70], [180, 255, 255]),
    'orange': ([11, 120, 70], [25, 255, 255]),
    'yellow': ([25, 120, 70], [35, 255, 255]),
    'green': ([36, 50, 50], [85, 255, 255]),
    'cyan': ([85, 100, 100], [95, 255, 255]),
    'blue': ([100, 50, 50], [130, 255, 255]),
    'purple': ([125, 50, 50], [150, 255, 255]),
    'pink': ([145, 50, 150], [170, 150, 255]),
    'brown': ([10, 100, 20], [20, 255, 150]),
    'black': ([0, 0, 0], [180, 255, 50]),
    'gray': ([0, 0, 100], [180, 50, 200]),
    'silver': ([0, 0, 150], [180, 30, 220]),
    'white': ([0, 0, 220], [180, 30, 255]),
}

def detect_advanced_color(img):
    if img is None or img.size == 0:
        return "unknown", 0.0
    
    try:
        h, w = img.shape[:2]
        if max(h, w) > 300:
            scale = 300 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        h, w = img.shape[:2]
        center_region = img[h//4:3*h//4, w//4:3*w//4]
        
        if center_region.size == 0:
            return "unknown", 0.0
        
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        color_scores = {}
        total_pixels = center_region.shape[0] * center_region.shape[1]
        
        for color_name, (lower, upper) in COLOR_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pixel_count = cv2.countNonZero(mask)
            score = pixel_count / total_pixels
            if score > 0.05:
                color_scores[color_name] = score
        
        if color_scores:
            best_color = max(color_scores, key=color_scores.get)
            confidence = color_scores[best_color]
            return best_color, confidence
        
        return "unknown", 0.0
    except:
        return "unknown", 0.0

# PLATE PREPROCESSING & OCR

def preprocess_plate_for_ocr(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None
    
    try:
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        
        h, w = gray.shape
        if w < 200:
            scale = 300 / w
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        elif w > 500:
            scale = 400 / w
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    except Exception as e:
        return plate_img

def detect_plate_region_contours(vehicle_img):
    if vehicle_img is None or vehicle_img.size == 0:
        return None
    
    try:
        h, w = vehicle_img.shape[:2]
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 30, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        plate_region = None
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                x, y, w_rect, h_rect = cv2.boundingRect(approx)
                aspect_ratio = w_rect / float(h_rect)
                
                if 1.5 < aspect_ratio < 6.0:
                    area = w_rect * h_rect
                    img_area = w * h
                    
                    if 0.01 < (area / img_area) < 0.3:
                        plate_region = vehicle_img[y:y+h_rect, x:x+w_rect]
                        break
        
        return plate_region
    except Exception as e:
        return None

ocr_reader = None
ocr_lock = threading.Lock()

def init_ocr():
    global ocr_reader
    
    if not ENABLE_PLATE_OCR:
        return
    
    try:
        if OCR_BACKEND == "easyocr":
            import easyocr
            ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
            print("✅ EasyOCR initialized")
        else:
            import pytesseract
            ocr_reader = "tesseract"
    except Exception as e:
        print(f"⚠️ OCR init failed: {e}")
        ocr_reader = None

def validate_indian_plate(text):
    if not text or len(text) < 6:
        return None, 0.0
    
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    patterns = [
        r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',
        r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$',
    ]
    
    for pattern in patterns:
        if re.match(pattern, cleaned):
            if len(cleaned) >= 9:
                formatted = f"{cleaned[:2]}-{cleaned[2:4]}-{cleaned[4:-4]}-{cleaned[-4:]}"
                return formatted, 0.9
    
    if any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned):
        if 6 <= len(cleaned) <= 12:
            return cleaned, 0.5
    
    return None, 0.0

def extract_plate_text_ultimate(vehicle_crop, plate_yolo_model=None):
    if ocr_reader is None or vehicle_crop is None:
        return "", 0.0, "none"
    
    plate_region = None
    method = "none"
    
    try:
        if USE_PLATE_YOLO and plate_yolo_model is not None:
            try:
                results = plate_yolo_model(vehicle_crop, verbose=False, conf=0.3)
                
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    best_idx = torch.argmax(boxes.conf).item()
                    box = boxes.xyxy[best_idx].cpu().numpy()
                    
                    x1, y1, x2, y2 = map(int, box)
                    plate_region = vehicle_crop[y1:y2, x1:x2]
                    method = "yolo"
            except Exception as e:
                pass
        
        if plate_region is None:
            plate_region = detect_plate_region_contours(vehicle_crop)
            if plate_region is not None:
                method = "contour"
        
        if plate_region is None:
            plate_region = vehicle_crop
            method = "direct"
        
        processed_plate = preprocess_plate_for_ocr(plate_region)
        
        if processed_plate is None:
            return "", 0.0, method
        
        with ocr_lock:
            if OCR_BACKEND == "easyocr":
                results = ocr_reader.readtext(processed_plate, detail=1, paragraph=False)
                
                if results:
                    plates = []
                    for (bbox, text, conf) in results:
                        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        
                        if 4 <= len(cleaned_text) <= 12:
                            has_digit = any(c.isdigit() for c in cleaned_text)
                            has_alpha = any(c.isalpha() for c in cleaned_text)
                            
                            if has_digit and has_alpha:
                                plates.append((cleaned_text, conf))
                    
                    if plates:
                        best_text, best_conf = max(plates, key=lambda x: x[1])
                        validated, val_conf = validate_indian_plate(best_text)
                        
                        if validated:
                            final_conf = (best_conf + val_conf) / 2
                            return validated, final_conf, method
                        elif best_conf > PLATE_MIN_CONFIDENCE:
                            return best_text, best_conf, method
            
            else:
                import pytesseract
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(processed_plate, config=custom_config)
                text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                if 4 <= len(text) <= 12:
                    validated, val_conf = validate_indian_plate(text)
                    if validated:
                        return validated, val_conf, method
                    elif any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
                        return text, 0.5, method
        
        return "", 0.0, method
    
    except Exception as e:
        return "", 0.0, method

# ========================================
# FACE RECOGNITION
# ========================================
known_face_encodings = []
known_face_names = []
face_pca = None
face_recognition_available = False

def init_face_recognition():
    global known_face_encodings, known_face_names, face_pca, face_recognition_available
    
    if not ENABLE_FACE_RECOGNITION:
        return
    
    try:
        import face_recognition
        import pickle
        
        if os.path.exists(FACE_ENCODINGS_FILE):
            with open(FACE_ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
            
            known_face_encodings = data.get('encodings', [])
            known_face_names = data.get('names', [])
            face_pca = data.get('pca', None)
            
            face_recognition_available = True
            print(f"✅ Face recognition loaded: {len(known_face_names)} faces")
        else:
            print(f"⚠️ Face encodings not found: {FACE_ENCODINGS_FILE}")
    except ImportError:
        print("⚠️ face_recognition not installed")
    except Exception as e:
        print(f"⚠️ Face recognition init error: {e}")

def recognize_face_async(person_crop):
    if not face_recognition_available or person_crop is None:
        return "Unknown", 0.0
    
    try:
        import face_recognition
        
        h, w = person_crop.shape[:2]
        if max(h, w) > 400:
            scale = 400 / max(h, w)
            person_crop = cv2.resize(person_crop, (int(w*scale), int(h*scale)))
        
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=0)
        
        if not face_locations:
            return "Unknown", 0.0
        
        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        
        if not face_encodings:
            return "Unknown", 0.0
        
        for face_encoding in face_encodings:
            if face_pca is not None:
                face_encoding_pca = face_pca.transform([face_encoding])[0]
                distances = np.linalg.norm(known_face_encodings - face_encoding_pca, axis=1)
            else:
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(distances) > 0:
                best_match_idx = np.argmin(distances)
                min_distance = distances[best_match_idx]
                
                if min_distance <= FACE_TOLERANCE:
                    confidence = 1.0 - (min_distance / FACE_TOLERANCE)
                    if confidence > 0.5:
                        return known_face_names[best_match_idx], confidence
        
        return "Unknown", 0.0
    except:
        return "Unknown", 0.0


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

def validate_human_presence(keypoints):
    """Validate human presence in keypoints."""
    if keypoints is None or len(keypoints) < 17:
        return False
    
    required_keypoints = [5, 6, 11, 12]
    min_keypoints_visible = 3
    
    valid_count = 0
    for i in required_keypoints:
        if (i < len(keypoints) and len(keypoints[i]) >= 2 and 
            keypoints[i][0] > 5 and keypoints[i][1] > 5):
            valid_count += 1
    
    return valid_count >= min_keypoints_visible

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
        
        padding = 20
        x_min = max(0, int(x_min) - padding)
        y_min = max(0, int(y_min) - padding)
        x_max = int(x_max) + padding
        y_max = int(y_max) + padding
        
        return [x_min, y_min, x_max, y_max]
    except:
        return None

# ========================================
# QUALITY ASSESSMENT
# ========================================
def calculate_sharpness(img):
    if img is None or img.size == 0:
        return 0.0
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except:
        return 0.0

def is_person_properly_visible(crop, box_width, box_height):
    """Check if person is properly visible (not just hand/leg)."""
    if crop is None or crop.size == 0:
        return False
    
    if box_width < MIN_PERSON_WIDTH or box_height < MIN_PERSON_HEIGHT:
        return False
    
    aspect_ratio = box_width / float(box_height)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False
    
    h, w = crop.shape[:2]
    center_region = crop[h//4:3*h//4, w//4:3*w//4]
    
    if center_region.size == 0:
        return False
    
    gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    
    if edge_density < 0.05:
        return False
    
    return True

# ========================================
# POSE TRACKER WITH ALERT
# ========================================
class PosePersonTracker:
    """Track person for pose analysis with alert system."""
    def __init__(self, person_id):
        self.id = person_id
        self.bbox = None
        self.keypoints = None
        self.pose = "standing"
        self.confidence = 0.0
        self.features = None
        self.last_seen = time.time()
        
        # Alert system
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

class MultiPoseTracker:
    """Manage multiple pose trackers."""
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
        """Update trackers with new detections."""
        matched = set()
        
        for bbox, keypoints, pose, confidence, features in detections:
            best_match_id = None
            best_iou = self.iou_threshold
            
            for tracker_id, tracker in self.trackers.items():
                if not tracker.is_active(current_time):
                    continue
                    
                if tracker.bbox is not None:
                    iou = self.calculate_iou(bbox, tracker.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = tracker_id
            
            if best_match_id is not None:
                self.trackers[best_match_id].update(bbox, keypoints, pose, confidence, features)
                matched.add(best_match_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.trackers[new_id] = PosePersonTracker(new_id)
                self.trackers[new_id].update(bbox, keypoints, pose, confidence, features)
                matched.add(new_id)
                print(f"👤 New pose person detected: ID {new_id}")
        
        inactive = [tid for tid, tracker in self.trackers.items() 
                   if not tracker.is_active(current_time)]
        for tid in inactive:
            del self.trackers[tid]
        
        return list(matched)
    
    def get_active_trackers(self, current_time):
        """Get all active trackers."""
        return {tid: tracker for tid, tracker in self.trackers.items() 
                if tracker.is_active(current_time)}

class PoseAlertSystem:
    """Alert system for pose detection."""
    def __init__(self):
        self.initial_threshold = LYING_ALERT_THRESHOLD
        self.max_alerts = MAX_ALERTS_PER_PERSON
        self.cooldown = ALERT_COOLDOWN
        self.min_consecutive_frames = MIN_CONSECUTIVE_FRAMES
        self.global_alert_count = 0
        
    def update_person(self, tracker, current_time):
        """Update alert state for a specific person."""
        if tracker.pose == 'lying_down':
            tracker.consecutive_lying_frames += 1
            
            if tracker.consecutive_lying_frames >= self.min_consecutive_frames:
                if tracker.lying_start_time is None:
                    tracker.lying_start_time = current_time
                    print(f"⏱️ Pose Person {tracker.id}: Lying timer started")
                
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

# ========================================
# VEHICLE & PERSON TRACKERS (OBJECT DETECTION)
# ========================================
class PersonTracker:
    def __init__(self, track_id):
        self.track_id = track_id
        self.best_frame = None
        self.best_quality = 0.0
        self.best_confidence = 0.0
        self.name = "Unknown"
        self.face_confidence = 0.0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.last_face_check = 0.0
        self.face_check_count = 0
        self.saved = False
        self.face_identified = False
        self.save_path = ""
        self.frame_count = 0
        self.good_frames_count = 0
        self.histogram = None
        
    def update(self, frame_crop, confidence, box_width, box_height):
        self.last_seen = time.time()
        self.frame_count += 1
        
        if not is_person_properly_visible(frame_crop, box_width, box_height):
            return False
        
        self.good_frames_count += 1
        quality = calculate_sharpness(frame_crop)
        
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_frame = frame_crop.copy()
            self.best_confidence = confidence
            
            self.histogram = cv2.calcHist([frame_crop], [0, 1, 2], None,
                                         [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(self.histogram, self.histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return True
        
        return False
    
    def should_check_face(self, current_time):
        """Only check face if not already identified and within attempt limits."""
        if self.face_identified:
            return False
        
        return (ENABLE_FACE_RECOGNITION and 
                FACE_RECOGNITION_ASYNC and
                current_time - self.last_face_check >= FACE_CHECK_INTERVAL and
                self.face_check_count < MAX_FACE_ATTEMPTS and
                self.best_frame is not None)
    
    def should_save(self):
        """Save only once when person is properly tracked."""
        duration = self.last_seen - self.first_seen
        
        return (not self.saved and 
                self.best_frame is not None and
                self.best_quality >= MIN_PERSON_QUALITY and
                self.good_frames_count >= PERSON_TRACKING_FRAMES and
                duration >= 1.0)

class VehicleTracker:
    def __init__(self, track_id, vehicle_type):
        self.track_id = track_id
        self.vehicle_type = vehicle_type
        self.best_frame = None
        self.best_quality = 0.0
        self.best_confidence = 0.0
        self.color = "unknown"
        self.color_confidence = 0.0
        self.plate_number = ""
        self.plate_confidence = 0.0
        self.plate_method = "none"
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.saved = False
        self.save_path = ""
        self.frame_count = 0
        self.histogram = None
        self.ocr_attempts = 0
        
    def update(self, frame_crop, confidence):
        self.last_seen = time.time()
        self.frame_count += 1
        
        quality = calculate_sharpness(frame_crop)
        
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_frame = frame_crop.copy()
            self.best_confidence = confidence
            
            color, color_conf = detect_advanced_color(frame_crop)
            if color_conf > self.color_confidence:
                self.color = color
                self.color_confidence = color_conf
            
            self.histogram = cv2.calcHist([frame_crop], [0, 1, 2], None, 
                                         [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(self.histogram, self.histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    def should_save(self):
        duration = self.last_seen - self.first_seen
        return (not self.saved and 
                self.best_quality >= MIN_VEHICLE_QUALITY and
                duration >= 1.0 and
                self.frame_count >= 3)

# ========================================
# DUPLICATE DETECTOR
# ========================================
class DuplicateDetector:
    def __init__(self):
        self.saved_vehicles = {}
        self.saved_persons = {}
        
    def is_duplicate_vehicle(self, tracker):
        if tracker.vehicle_type not in self.saved_vehicles:
            return False
        
        if tracker.histogram is None:
            return False
        
        current_time = time.time()
        
        for saved_hist, saved_time, saved_id in self.saved_vehicles[tracker.vehicle_type]:
            if current_time - saved_time < MIN_TIME_BETWEEN_SAME_VEHICLE:
                similarity = cv2.compareHist(tracker.histogram, saved_hist, cv2.HISTCMP_CORREL)
                
                if similarity >= VEHICLE_SIMILARITY_THRESHOLD:
                    print(f"📄 Duplicate vehicle: ID {tracker.track_id} ≈ ID {saved_id} ({similarity:.2f})")
                    return True
        
        return False
    
    def is_duplicate_person(self, tracker):
        if tracker.name not in self.saved_persons:
            return False
        
        if tracker.histogram is None:
            return False
        
        current_time = time.time()
        
        for saved_hist, saved_time, saved_id in self.saved_persons[tracker.name]:
            if current_time - saved_time < MIN_TIME_BETWEEN_SAME_VEHICLE:
                similarity = cv2.compareHist(tracker.histogram, saved_hist, cv2.HISTCMP_CORREL)
                
                if similarity >= PERSON_SIMILARITY_THRESHOLD:
                    print(f"📄 Duplicate person: ID {tracker.track_id} ({tracker.name}) ≈ ID {saved_id} ({similarity:.2f})")
                    return True
        
        return False
    
    def add_vehicle(self, tracker):
        if tracker.vehicle_type not in self.saved_vehicles:
            self.saved_vehicles[tracker.vehicle_type] = []
        
        self.saved_vehicles[tracker.vehicle_type].append(
            (tracker.histogram.copy(), time.time(), tracker.track_id)
        )
        
        self.saved_vehicles[tracker.vehicle_type] = [
            (h, t, tid) for h, t, tid in self.saved_vehicles[tracker.vehicle_type]
            if time.time() - t < 60
        ]
    
    def add_person(self, tracker):
        if tracker.name not in self.saved_persons:
            self.saved_persons[tracker.name] = []
        
        self.saved_persons[tracker.name].append(
            (tracker.histogram.copy(), time.time(), tracker.track_id)
        )
        
        self.saved_persons[tracker.name] = [
            (h, t, tid) for h, t, tid in self.saved_persons[tracker.name]
            if time.time() - t < 60
        ]

# ========================================
# UNIFIED DETECTION SYSTEM
# ========================================
class UnifiedDetectionSystem:
    def __init__(self):
        print(f"\n{'='*60}")
        print("🚀 UNIFIED DETECTION SYSTEM v3.0")
        print(f"{'='*60}\n")
        
        # Load YOLO models
        print(f"📦 Loading Object Detection Model: {YOLO_OBJECT_MODEL}")
        self.object_model = YOLO(YOLO_OBJECT_MODEL)
        self.object_model.to(DEVICE)
        print(f"✅ Object model loaded on {DEVICE}\n")
        
        if POSE_ALERT_ENABLED:
            print(f"📦 Loading Pose Detection Model: {YOLO_POSE_MODEL}")
            self.pose_model = YOLO(YOLO_POSE_MODEL)
            self.pose_model.to(DEVICE)
            print(f"✅ Pose model loaded on {DEVICE}\n")
        else:
            self.pose_model = None
        
        # Plate model
        self.plate_model = None
        if USE_PLATE_YOLO:
            try:
                if os.path.exists(PLATE_YOLO_MODEL):
                    print(f"📦 Loading Plate Model: {PLATE_YOLO_MODEL}")
                    self.plate_model = YOLO(PLATE_YOLO_MODEL)
                    self.plate_model.to(DEVICE)
                    print(f"✅ Plate model loaded\n")
            except Exception as e:
                print(f"⚠️ Plate model load error: {e}\n")
        
        # Initialize components
        init_ocr()
        init_face_recognition()
        
        # Thread pools
        self.face_executor = ThreadPoolExecutor(max_workers=MAX_FACE_WORKERS)
        self.ocr_executor = ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS)
        
        # Trackers
        self.active_vehicles = {}
        self.active_persons = {}
        self.pose_tracker = MultiPoseTracker() if POSE_ALERT_ENABLED else None
        self.pose_alert_system = PoseAlertSystem() if POSE_ALERT_ENABLED else None
        self.duplicate_detector = DuplicateDetector()
        
        # Statistics
        self.stats = {
            'vehicles_saved': 0,
            'persons_saved': 0,
            'faces_recognized': 0,
            'plates_detected': 0,
            'plates_yolo': 0,
            'plates_contour': 0,
            'plates_direct': 0,
            'duplicates_prevented': 0,
            'partial_persons_skipped': 0,
            'pose_alerts': 0,
            'pose_people_tracked': 0
        }
        
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        print(f"{'='*60}")
        print("✅ INITIALIZATION COMPLETE")
        print(f"{'='*60}\n")
    
    def save_vehicle(self, tracker):
        """Save vehicle detection."""
        if tracker.saved or tracker.best_frame is None:
            return
        
        if self.duplicate_detector.is_duplicate_vehicle(tracker):
            tracker.saved = True
            self.stats['duplicates_prevented'] += 1
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vehicle_{tracker.track_id}_{tracker.vehicle_type}_{timestamp}.jpg"
            save_path = os.path.join(SAVE_DIR, "vehicles", filename)
            
            cv2.imwrite(save_path, tracker.best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            tracker.save_path = save_path
            tracker.saved = True
            
            if ENABLE_PLATE_OCR and not tracker.plate_number and tracker.ocr_attempts < 3:
                tracker.ocr_attempts += 1
                plate_text, plate_conf, method = extract_plate_text_ultimate(
                    tracker.best_frame, self.plate_model
                )
                
                if plate_text and plate_conf > PLATE_MIN_CONFIDENCE:
                    tracker.plate_number = plate_text
                    tracker.plate_confidence = plate_conf
                    tracker.plate_method = method
                    self.stats['plates_detected'] += 1
                    
                    if method == "yolo":
                        self.stats['plates_yolo'] += 1
                    elif method == "contour":
                        self.stats['plates_contour'] += 1
                    else:
                        self.stats['plates_direct'] += 1
                    
                    plate_filename = f"plate_{tracker.track_id}_{timestamp}.jpg"
                    plate_path = os.path.join(SAVE_DIR, "plates", plate_filename)
                    cv2.imwrite(plate_path, tracker.best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    append_csv(PLATE_CSV, [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        tracker.track_id,
                        plate_text,
                        f"{plate_conf:.3f}",
                        tracker.vehicle_type,
                        method
                    ])
            
            append_csv(VEHICLE_CSV, [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tracker.track_id,
                tracker.vehicle_type,
                tracker.color,
                f"{tracker.best_confidence:.3f}",
                tracker.plate_number,
                f"{tracker.best_quality:.1f}",
                save_path,
                datetime.fromtimestamp(tracker.first_seen).strftime("%Y-%m-%d %H:%M:%S"),
                datetime.fromtimestamp(tracker.last_seen).strftime("%Y-%m-%d %H:%M:%S")
            ])
            
            self.duplicate_detector.add_vehicle(tracker)
            self.stats['vehicles_saved'] += 1
            
            plate_info = f"{tracker.plate_number} [{tracker.plate_method}]" if tracker.plate_number else "N/A"
            print(f"💾 VEHICLE: ID {tracker.track_id} | {tracker.vehicle_type} | {tracker.color} | Plate: {plate_info}")
            
        except Exception as e:
            print(f"⚠️ Error saving vehicle {tracker.track_id}: {e}")
    
    def save_person(self, tracker):
        """Save person detection."""
        if tracker.saved or tracker.best_frame is None:
            return
        
        if self.duplicate_detector.is_duplicate_person(tracker):
            tracker.saved = True
            self.stats['duplicates_prevented'] += 1
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_safe = tracker.name.replace(' ', '_') if tracker.name != "Unknown" else "unknown"
            filename = f"person_{tracker.track_id}_{name_safe}_{timestamp}.jpg"
            save_path = os.path.join(SAVE_DIR, "persons", filename)
            
            cv2.imwrite(save_path, tracker.best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            tracker.save_path = save_path
            tracker.saved = True
            
            if tracker.name != "Unknown":
                face_filename = f"face_{tracker.track_id}_{name_safe}_{timestamp}.jpg"
                face_path = os.path.join(SAVE_DIR, "faces", face_filename)
                cv2.imwrite(face_path, tracker.best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            append_csv(PERSON_CSV, [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tracker.track_id,
                tracker.name,
                f"{tracker.face_confidence:.3f}",
                f"{tracker.best_quality:.1f}",
                save_path,
                datetime.fromtimestamp(tracker.first_seen).strftime("%Y-%m-%d %H:%M:%S"),
                datetime.fromtimestamp(tracker.last_seen).strftime("%Y-%m-%d %H:%M:%S"),
                tracker.face_check_count
            ])
            
            self.duplicate_detector.add_person(tracker)
            self.stats['persons_saved'] += 1
            
            if tracker.name != "Unknown":
                self.stats['faces_recognized'] += 1
            
            print(f"💾 PERSON: ID {tracker.track_id} | {tracker.name} | Q: {tracker.best_quality:.1f}")
            
        except Exception as e:
            print(f"⚠️ Error saving person {tracker.track_id}: {e}")
    
    def handle_face_recognition_result(self, future, tracker):
        """Handle face recognition result."""
        try:
            name, confidence = future.result(timeout=3.0)
            
            if name != "Unknown" and confidence > tracker.face_confidence:
                tracker.name = name
                tracker.face_confidence = confidence
                tracker.face_identified = True
                
                append_csv(FACE_CSV, [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    tracker.track_id,
                    name,
                    f"{confidence:.3f}",
                    f"{tracker.best_quality:.1f}"
                ])
                
                print(f"👤 FACE: ID {tracker.track_id} → {name} ({confidence:.3f})")
            elif confidence > tracker.face_confidence:
                tracker.face_confidence = confidence
        except:
            pass
    
    def save_pose_alert_image(self, frame, person_id, duration, alert_count):
        """Save pose alert image with annotation."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_alert_{alert_count}_person{person_id}_{timestamp}.jpg"
            image_path = os.path.join(SAVE_DIR, "pose_alerts", filename)
            
            annotated_frame = frame.copy()
            
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (5, 5), (450, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
            
            cv2.putText(annotated_frame, f"POSE ALERT #{alert_count}", (15, 35), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Person ID: {person_id} | Duration: {duration:.1f}s", (15, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                       (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imwrite(image_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"💾 Pose alert image saved: {filename}")
            return image_path
            
        except Exception as e:
            print(f"⚠️ Save pose alert error: {str(e)[:50]}")
            return ""
    
    def cleanup_inactive_trackers(self, current_time):
        """Cleanup inactive object detection trackers."""
        timeout = 5.0
        
        inactive_vehicles = [tid for tid, tracker in self.active_vehicles.items() 
                            if current_time - tracker.last_seen > timeout]
        
        for tid in inactive_vehicles:
            tracker = self.active_vehicles[tid]
            if tracker.should_save():
                self.save_vehicle(tracker)
            del self.active_vehicles[tid]
        
        inactive_persons = [tid for tid, tracker in self.active_persons.items()
                           if current_time - tracker.last_seen > timeout]
        
        for tid in inactive_persons:
            tracker = self.active_persons[tid]
            if tracker.should_save():
                self.save_person(tracker)
            del self.active_persons[tid]
    
    def process_video(self, video_path):
        """Main video processing loop."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"🎬 VIDEO PROCESSING STARTED")
        print(f"{'='*60}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.1f}")
        print(f"Total Frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.1f}s")
        print(f"Object Detection: ENABLED")
        print(f"Pose Detection: {'ENABLED' if POSE_ALERT_ENABLED else 'DISABLED'}")
        print(f"Notifications: {'ENABLED' if (TELEGRAM_ENABLED or EMAIL_ENABLED) else 'DISABLED'}")
        print(f"{'='*60}\n")
        
        if SHOW_VIDEO:
            cv2.namedWindow("Unified Detection System", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Unified Detection System", DISPLAY_WIDTH, int(DISPLAY_WIDTH * height / width))
        
        frame_id = 0
        start_time = time.time()
        last_fps_time = start_time
        fps_counter = 0
        display_fps = 0.0
        last_cleanup = start_time
        last_pose_log = defaultdict(float)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\n✅ Video processing completed")
                    break
                
                frame_id += 1
                current_time = time.time()
                
                fps_counter += 1
                if current_time - last_fps_time >= 1.0:
                    display_fps = fps_counter / (current_time - last_fps_time)
                    fps_counter = 0
                    last_fps_time = current_time
                
                display_frame = frame.copy() if SHOW_VIDEO else None
                
                # ===== OBJECT DETECTION (Vehicles & Persons) =====
                object_results = self.object_model.track(
                    source=frame,
                    conf=OBJECT_DETECTION_CONFIDENCE,
                    iou=IOU_THRESHOLD,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                    classes=[0, 2, 3, 5, 7],
                    max_det=MAX_DETECTIONS,
                    device=DEVICE
                )
                
                if object_results and object_results[0].boxes is not None and len(object_results[0].boxes) > 0:
                    boxes = object_results[0].boxes
                    
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        if hasattr(boxes, 'id') and boxes.id is not None and i < len(boxes.id):
                            track_id = int(boxes.id[i].cpu().numpy())
                        else:
                            track_id = frame_id * 1000 + i
                        
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        box_width = x2 - x1
                        box_height = y2 - y1
                        area = box_width * box_height
                        
                        crop = frame[y1:y2, x1:x2].copy()
                        
                        if crop is None or crop.size == 0:
                            continue
                        
                        # PERSON PROCESSING
                        if cls_id == 0 and area >= MIN_PERSON_AREA:
                            if track_id not in self.active_persons:
                                self.active_persons[track_id] = PersonTracker(track_id)
                                print(f"👤 New person detected: ID {track_id}")
                            
                            tracker = self.active_persons[track_id]
                            updated = tracker.update(crop, conf, box_width, box_height)
                            
                            if not updated:
                                self.stats['partial_persons_skipped'] += 1
                            
                            if not tracker.face_identified and tracker.should_check_face(current_time):
                                tracker.last_face_check = current_time
                                tracker.face_check_count += 1
                                
                                future = self.face_executor.submit(
                                    recognize_face_async,
                                    tracker.best_frame.copy()
                                )
                                future.add_done_callback(
                                    lambda f, t=tracker: self.handle_face_recognition_result(f, t)
                                )
                            
                            if tracker.should_save():
                                self.save_person(tracker)
                            
                            if SHOW_VIDEO:
                                if tracker.saved:
                                    color = (0, 255, 255)
                                elif tracker.face_identified:
                                    color = (0, 255, 0)
                                else:
                                    color = (255, 255, 0)
                                
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                
                                label = f"ID:{track_id}"
                                if tracker.face_identified:
                                    label += f" | {tracker.name} ✓"
                                elif tracker.name != "Unknown":
                                    label += f" | {tracker.name}"
                                
                                label += f" | Q:{tracker.best_quality:.0f}"
                                if tracker.saved:
                                    label += " [SAVED]"
                                
                                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(display_frame, (x1, y1 - label_h - 10), 
                                            (x1 + label_w, y1), color, -1)
                                cv2.putText(display_frame, label, (x1, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        
                        # VEHICLE PROCESSING
                        elif cls_id in self.vehicle_classes and area >= MIN_VEHICLE_AREA:
                            vehicle_type = self.vehicle_classes[cls_id]
                            
                            if track_id not in self.active_vehicles:
                                self.active_vehicles[track_id] = VehicleTracker(track_id, vehicle_type)
                                print(f"🚗 New vehicle: ID {track_id} ({vehicle_type})")
                            
                            tracker = self.active_vehicles[track_id]
                            tracker.update(crop, conf)
                            
                            if tracker.should_save():
                                self.save_vehicle(tracker)
                            
                            if SHOW_VIDEO:
                                color = (0, 255, 255) if tracker.saved else (255, 0, 255)
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                
                                label = f"ID:{track_id} | {vehicle_type}"
                                if tracker.color != "unknown":
                                    label += f" | {tracker.color}"
                                if tracker.plate_number:
                                    label += f" | {tracker.plate_number}"
                                if tracker.saved:
                                    label += " [SAVED]"
                                
                                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(display_frame, (x1, y1 - label_h - 10),
                                            (x1 + label_w, y1), color, -1)
                                cv2.putText(display_frame, label, (x1, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                # ===== POSE DETECTION =====
                if POSE_ALERT_ENABLED and self.pose_model is not None:
                    pose_results = self.pose_model(
                        frame, 
                        verbose=False, 
                        conf=POSE_DETECTION_CONFIDENCE,
                        imgsz=YOLO_IMG_SIZE,
                        device=DEVICE
                    )
                    
                    pose_detections = []
                    
                    if (len(pose_results) > 0 and 
                        pose_results[0].keypoints is not None and 
                        len(pose_results[0].keypoints.xy) > 0):
                        
                        all_keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                        
                        for person_idx, keypoints in enumerate(all_keypoints):
                            if validate_human_presence(keypoints):
                                features = extract_pose_features(keypoints)
                                
                                if features is not None:
                                    pose_label, confidence_scores, _ = classify_pose_accurate(
                                        features, "standing"
                                    )
                                    
                                    main_confidence = max(confidence_scores.values())
                                    bbox = get_person_bbox(keypoints)
                                    
                                    if bbox is not None and pose_label != "uncertain":
                                        pose_detections.append((bbox, keypoints, pose_label, 
                                                              main_confidence, features))
                    
                    # Update pose trackers
                    active_pose_ids = self.pose_tracker.update(pose_detections, current_time)
                    active_pose_trackers = self.pose_tracker.get_active_trackers(current_time)
                    
                    self.stats['pose_people_tracked'] = len(active_pose_trackers)
                    
                    # Process each pose tracker
                    for pose_tracker_id, pose_tracker in active_pose_trackers.items():
                        # Draw skeleton
                        if SHOW_VIDEO:
                            display_frame = draw_complete_skeleton(
                                display_frame, pose_tracker.keypoints, pose_tracker.pose
                            )
                            
                            # Draw pose bounding box
                            bbox = pose_tracker.bbox
                            if bbox is not None:
                                if pose_tracker.pose == 'lying_down':
                                    box_color = (0, 0, 255)
                                elif pose_tracker.pose == 'sitting':
                                    box_color = (255, 255, 0)
                                else:
                                    box_color = (0, 255, 0)
                                
                                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)
                                
                                label = f"POSE-ID:{pose_tracker_id} | {pose_tracker.pose.upper()}"
                                label += f" | {pose_tracker.confidence:.2f}"
                                
                                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(display_frame, (bbox[0], bbox[1] - label_h - 10),
                                            (bbox[0] + label_w, bbox[1]), box_color, -1)
                                cv2.putText(display_frame, label, (bbox[0], bbox[1] - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Log to CSV periodically
                        if (current_time - last_pose_log[pose_tracker_id] >= 2.0 and 
                            pose_tracker.pose != "uncertain" and pose_tracker.features is not None):
                            
                            append_csv(POSE_CSV, [
                                current_time,
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                pose_tracker.pose,
                                pose_tracker_id,
                                f"{pose_tracker.confidence:.3f}",
                                f"{pose_tracker.features['torso_angle']:.1f}",
                                f"{pose_tracker.features['avg_knee_angle']:.1f}",
                                f"{pose_tracker.features['avg_hip_angle']:.1f}",
                                False,
                                0,
                                pose_tracker.alert_count
                            ])
                            
                            last_pose_log[pose_tracker_id] = current_time
                        
                        # Check for pose alerts
                        if pose_tracker.pose == 'lying_down' and pose_tracker.confidence > 0.55:
                            alert_triggered, lying_duration, alert_count, person_id = \
                                self.pose_alert_system.update_person(pose_tracker, current_time)
                            
                            if alert_triggered:
                                alert_image_path = self.save_pose_alert_image(
                                    display_frame if SHOW_VIDEO else frame, 
                                    person_id, lying_duration, alert_count
                                )
                                
                                self.stats['pose_alerts'] += 1
                                
                                print(f"\n{'='*60}")
                                print(f"🚨 POSE ALERT #{alert_count} - PERSON {person_id}")
                                print(f"⏱️  Duration: {lying_duration:.1f}s")
                                print(f"🎯 Confidence: {pose_tracker.confidence:.2f}")
                                print(f"{'='*60}\n")
                                
                                # Log alert
                                append_csv(POSE_CSV, [
                                    current_time,
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'lying_down',
                                    person_id,
                                    f"{pose_tracker.confidence:.3f}",
                                    f"{pose_tracker.features['torso_angle']:.1f}",
                                    f"{pose_tracker.features['avg_knee_angle']:.1f}",
                                    f"{pose_tracker.features['avg_hip_angle']:.1f}",
                                    True,
                                    f"{lying_duration:.1f}",
                                    alert_count
                                ])
                                
                                # Send alert
                                alert_message = f"""🚨 POSE ALERT #{alert_count}

👤 Person ID: {person_id}
⏰ Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
⏱️ Duration: {lying_duration:.1f}s
🎯 Confidence: {pose_tracker.confidence:.2f}
📍 Status: Person lying down

⚠️ Please check immediately!"""
                                
                                send_alert_ultra_fast(alert_message, alert_image_path)
                
                # Display info overlay
                if SHOW_VIDEO and display_frame is not None:
                    overlay_h = 160
                    overlay = np.zeros((overlay_h, width, 3), dtype=np.uint8)
                    
                    progress = (frame_id / total_frames) * 100 if total_frames > 0 else 0
                    
                    info_lines = [
                        f"Frame: {frame_id}/{total_frames} ({progress:.1f}%) | FPS: {display_fps:.1f}",
                        f"Objects: V:{len(self.active_vehicles)} P:{len(self.active_persons)}",
                        f"Saved: Vehicles:{self.stats['vehicles_saved']} Persons:{self.stats['persons_saved']} Faces:{self.stats['faces_recognized']}",
                        f"Plates: {self.stats['plates_detected']} | Duplicates Prevented: {self.stats['duplicates_prevented']}"
                    ]
                    
                    if POSE_ALERT_ENABLED:
                        info_lines.append(f"Pose: People:{self.stats['pose_people_tracked']} Alerts:{self.stats['pose_alerts']}")
                    
                    for i, line in enumerate(info_lines):
                        cv2.putText(overlay, line, (10, 25 + i * 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    display_frame[0:overlay_h] = cv2.addWeighted(
                        display_frame[0:overlay_h], 0.3, overlay, 0.7, 0
                    )
                    
                    display_h = int(DISPLAY_WIDTH * height / width)
                    display_resized = cv2.resize(display_frame, (DISPLAY_WIDTH, display_h))
                    cv2.imshow("Unified Detection System", display_resized)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        print("\n⏸️  Processing stopped by user")
                        break
                    elif key == ord('s'):  # Screenshot
                        screenshot_path = os.path.join(SAVE_DIR, f"screenshot_{frame_id}.jpg")
                        cv2.imwrite(screenshot_path, display_frame)
                        print(f"📸 Screenshot saved: {screenshot_path}")
                
                # Periodic cleanup
                if current_time - last_cleanup >= 5.0:
                    self.cleanup_inactive_trackers(current_time)
                    last_cleanup = current_time
                
                # Progress updates
                if frame_id % 100 == 0:
                    elapsed = current_time - start_time
                    estimated_total = (elapsed / frame_id) * total_frames if frame_id > 0 else 0
                    remaining = estimated_total - elapsed
                    
                    print(f"⏱️  Progress: {frame_id}/{total_frames} ({progress:.1f}%) | "
                          f"Elapsed: {elapsed:.0f}s | Remaining: ~{remaining:.0f}s | "
                          f"FPS: {display_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Final cleanup
            print("\n🔄 Performing final cleanup...")
            current_time = time.time()
            
            for tracker in self.active_vehicles.values():
                if tracker.should_save():
                    self.save_vehicle(tracker)
            
            for tracker in self.active_persons.values():
                if tracker.should_save():
                    self.save_person(tracker)
            
            cap.release()
            if SHOW_VIDEO:
                cv2.destroyAllWindows()
            
            # Shutdown executors
            self.face_executor.shutdown(wait=False)
            self.ocr_executor.shutdown(wait=False)
            
            # Final statistics
            elapsed = time.time() - start_time
            avg_fps = frame_id / elapsed if elapsed > 0 else 0
            
            print(f"\n{'='*60}")
            print("📊 FINAL STATISTICS")
            print(f"{'='*60}")
            print(f"Total Frames Processed: {frame_id}")
            print(f"Processing Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"\n🚗 Vehicles:")
            print(f"  - Saved: {self.stats['vehicles_saved']}")
            print(f"  - Plates Detected: {self.stats['plates_detected']}")
            print(f"    • YOLO: {self.stats['plates_yolo']}")
            print(f"    • Contour: {self.stats['plates_contour']}")
            print(f"    • Direct: {self.stats['plates_direct']}")
            print(f"\n👤 Persons:")
            print(f"  - Saved: {self.stats['persons_saved']}")
            print(f"  - Faces Recognized: {self.stats['faces_recognized']}")
            print(f"  - Partial Detections Skipped: {self.stats['partial_persons_skipped']}")
            
            if POSE_ALERT_ENABLED:
                print(f"\n🧍 Pose Detection:")
                print(f"  - People Tracked: {self.stats['pose_people_tracked']}")
                print(f"  - Alerts Triggered: {self.stats['pose_alerts']}")
            
            print(f"\n📄 Duplicates Prevented: {self.stats['duplicates_prevented']}")
            print(f"\n💾 Output Directory: {SAVE_DIR}")
            print(f"{'='*60}\n")

# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    try:
        system = UnifiedDetectionSystem()
        system.process_video(VIDEO_PATH)
        
        print("\n✅ All operations completed successfully!")
        print(f"📂 Check output at: {SAVE_DIR}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n👋 Goodbye!")