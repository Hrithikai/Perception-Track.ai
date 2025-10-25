# optimized_person_saving.py
# FIXED VERSION - Saves best person image for every tracked person

import os, time, json, cv2, numpy as np, threading, pickle
import torch
from datetime import datetime
from ultralytics import YOLO
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ========== OPTIMIZED CONFIG ==========
VIDEO_PATH = "D:\Perception Track.ai\dataset\input video for vechicals\VID_20250915_103941962.mp4"
MODEL_WEIGHTS = "yolov8s.pt"
PROCESSING_WIDTH = 640
DETECTION_SKIP = 4
DEVICE = "cuda"
USE_FP16 = True
MAX_FACE_WORKERS = 2
MAX_OCR_WORKERS = 1
SAVE_DIR = r"D:\Perception Track.ai\up_optimized m"
ENCODINGS_FILE = "face_encodings_pca.pkl"
TOLERANCE = 0.5

# Performance optimizations
MIN_VEHICLE_AREA = 6000
MIN_PERSON_AREA = 5000
FACE_MIN_SHARPNESS = 10
TRACK_BUFFER_SIZE = 20
GPU_MEMORY_FRACTION = 0.7

# Face tracking
FACE_TRACKING_INTERVAL = 1.0
FACE_QUALITY_THRESHOLD = 8

# Visualization
SHOW_VIDEO = True
WINDOW_NAME = "Person Saving Fixed"
VIDEO_WIDTH = 1280

# Quality thresholds
MIN_VEHICLE_QUALITY = 30
MIN_TRACK_DURATION = 1.5
PERSON_SAVE_QUALITY = 15  # Quality threshold to save person image

# ========== SETUP DIRECTORIES ==========
for subdir in ["vehicles", "persons", "best_faces", "screenshots", "person_crops"]:
    os.makedirs(os.path.join(SAVE_DIR, subdir), exist_ok=True)

# ========== LOAD FACE RECOGNITION ==========
print("Loading face recognition model...")
try:
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    known_face_encodings_pca = data["encodings"]
    known_face_names = data["names"]
    pca = data["pca"]
    print(f"✅ Loaded {len(known_face_names)} face encodings")
except Exception as e:
    print(f"❌ Error loading face encodings: {e}")
    known_face_encodings_pca = np.array([])
    known_face_names = []
    pca = None

# ========== OPTIMIZED UTILITY FUNCTIONS ==========
def now_ts(): 
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def fast_sharpness(img):
    if img is None or img.size == 0: 
        return 0.0
    if img.ndim == 3: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(img, cv2.CV_64F, ksize=3).var())

def get_dominant_color_fast(img):
    if img is None or img.size == 0:
        return "unknown"
    
    h, w = img.shape[:2]
    center_region = img[h//3:2*h//3, w//3:2*w//3]
    
    if center_region.size == 0:
        return "unknown"
        
    hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:,:,0])
    avg_sat = np.mean(hsv[:,:,1])
    avg_val = np.mean(hsv[:,:,2])
    
    if avg_val < 60: return "black"
    elif avg_val > 200 and avg_sat < 30: return "white"
    elif avg_sat < 30: return "gray" if 100 < avg_val < 200 else "silver"
    elif avg_hue < 10 or avg_hue > 170: return "red"
    elif 10 <= avg_hue < 25: return "orange"
    elif 25 <= avg_hue < 35: return "yellow" 
    elif 35 <= avg_hue < 85: return "green"
    elif 85 <= avg_hue < 130: return "blue"
    else: return "unknown"

# ========== OPTIMIZED VISUALIZATION ==========
def draw_fast_detection(frame, x1, y1, x2, y2, track_id, class_name, confidence, color, name=""):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    label = f"ID:{track_id} {class_name}"
    if name and name != "Unknown":
        label += f" | {name}"
    
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def add_fast_status_bar(frame, frame_id, total_frames, fps, vehicles_count, persons_count, face_tracks):
    h, w = frame.shape[:2]
    status_bar = np.zeros((40, w, 3), dtype=np.uint8)
    
    progress = (frame_id / total_frames) * 100 if total_frames > 0 else 0
    status_texts = [
        f"Frame: {frame_id}/{total_frames} ({progress:.1f}%)",
        f"FPS: {fps:.1f}",
        f"Vehicles: {vehicles_count}",
        f"Persons: {persons_count}",
        f"Face Tracks: {face_tracks}"
    ]
    
    for i, text in enumerate(status_texts):
        x_pos = 10 + i * 200
        cv2.putText(status_bar, text, (x_pos, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    tracking_info = f"Face Tracking: Every {FACE_TRACKING_INTERVAL}s | Quality: >{FACE_QUALITY_THRESHOLD}"
    cv2.putText(status_bar, tracking_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    frame[0:40, 0:w] = status_bar

def get_fast_color(track_id):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    return colors[track_id % len(colors)]

# ========== CSV HANDLING ==========
csv_lock = threading.Lock()

def append_fast_csv(path, row):
    with csv_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(",".join([str(x).replace(",", "") for x in row]) + "\n")

# ========== MODEL INITIALIZATION ==========
print(f"🚀 Initializing YOLO on {DEVICE}...")

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    torch.cuda.empty_cache()

device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
print(f"Using device: {device}")

print("Loading YOLO model...")
try:
    yolo = YOLO(MODEL_WEIGHTS)
    yolo.to(device)
    print(f"✅ YOLO model loaded on {device}")
except Exception as e:
    print(f"❌ Error: {e}")
    device = torch.device("cpu")
    yolo = YOLO(MODEL_WEIGHTS)
    yolo.to(device)

# ========== THREAD POOLS ==========
face_executor = ThreadPoolExecutor(max_workers=MAX_FACE_WORKERS)

# ========== OPTIMIZED TRACKING CONTAINERS ==========
class FastPersonTracker:
    def __init__(self, track_id):
        self.track_id = track_id
        self.label = f"Person_{track_id}"
        self.best_frame = None
        self.best_quality = 0
        self.current_frame = None
        self.current_quality = 0
        self.name = "Unknown"
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.last_face_track = 0
        self.face_tracking_count = 0
        self.best_face_confidence = 0.0
        self.best_face_frame = None
        self.saved = False
        self.best_frame_saved = False  # Track if best frame is saved

class FastVehicleTracker:
    def __init__(self, track_id):
        self.track_id = track_id
        self.best_frame = None
        self.best_quality = 0
        self.vehicle_type = "unknown"
        self.color = "unknown"
        self.saved = False
        self.last_seen = time.time()
        self.first_seen = time.time()

active_vehicles = {}
active_persons = {}
saved_vehicle_ids = set()
saved_person_ids = set()

# Create CSV files
face_csv = os.path.join(SAVE_DIR, "face_tracking.csv")
vehicle_csv = os.path.join(SAVE_DIR, "vehicles.csv")
person_csv = os.path.join(SAVE_DIR, "persons.csv")

# Initialize CSV headers
for path, header in [(face_csv, ["timestamp", "track_id", "name", "confidence", "quality"]),
                     (vehicle_csv, ["timestamp", "track_id", "vehicle_type", "color", "image_path"]),
                     (person_csv, ["timestamp", "track_id", "name", "best_confidence", "image_path", "quality"])]:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")

# ========== COORDINATE MAPPING ==========
def fast_map_coords(box, small_w, small_h, orig_w, orig_h):
    x1, y1, x2, y2 = box
    scale_x = orig_w / small_w
    scale_y = orig_h / small_h
    return (int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y))

# ========== PERSON SAVING FUNCTIONS ==========
def save_best_person_image(person_tracker):
    """Save the best quality image of the person"""
    if person_tracker.best_frame_saved or person_tracker.best_frame is None:
        return
    
    # Only save if quality is good enough
    if person_tracker.best_quality < PERSON_SAVE_QUALITY:
        return
    
    try:
        # Save the best person image
        person_filename = f"person_{person_tracker.track_id}_best_{int(person_tracker.best_quality)}.jpg"
        person_path = os.path.join(SAVE_DIR, "persons", person_filename)
        
        cv2.imwrite(person_path, person_tracker.best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Update CSV
        append_fast_csv(person_csv, [
            now_ts(), person_tracker.track_id, 
            person_tracker.name, person_tracker.best_face_confidence,
            person_path, person_tracker.best_quality
        ])
        
        person_tracker.best_frame_saved = True
        saved_person_ids.add(person_tracker.track_id)
        
        print(f"💾 SAVED PERSON: ID {person_tracker.track_id} (quality: {person_tracker.best_quality:.1f})")
        
    except Exception as e:
        print(f"❌ Error saving person {person_tracker.track_id}: {e}")

def save_person_always(person_tracker):
    """Save person regardless of face recognition"""
    if person_tracker.track_id in saved_person_ids:
        return
    
    # Save even if no face recognition, but with best available image
    if person_tracker.best_frame is not None:
        person_filename = f"person_{person_tracker.track_id}_detected.jpg"
        person_path = os.path.join(SAVE_DIR, "persons", person_filename)
        
        cv2.imwrite(person_path, person_tracker.best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        append_fast_csv(person_csv, [
            now_ts(), person_tracker.track_id, 
            person_tracker.name, person_tracker.best_face_confidence,
            person_path, person_tracker.best_quality
        ])
        
        saved_person_ids.add(person_tracker.tracker_id)
        print(f"💾 SAVED PERSON (no face): ID {person_tracker.track_id}")

# ========== ULTRA-FAST FACE RECOGNITION ==========
def ultra_fast_face_recognition(person_crop, person_tracker, is_tracking=False):
    result = {
        "track_id": person_tracker.track_id,
        "name": "Unknown",
        "confidence": 0.0,
        "quality": person_tracker.current_quality,
        "faces_detected": 0,
        "timestamp": now_ts()
    }
    
    try:
        if person_crop is None or person_crop.size == 0:
            return result
        
        import face_recognition
        
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        h, w = rgb_crop.shape[:2]
        
        if max(h, w) > 400:
            scale = 400 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            rgb_crop = cv2.resize(rgb_crop, (new_w, new_h))
        
        face_locations = face_recognition.face_locations(rgb_crop, model="hog", number_of_times_to_upsample=0)
        result["faces_detected"] = len(face_locations)
        
        if not face_locations:
            return result
        
        face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)
        
        if face_encodings and len(known_face_encodings_pca) > 0:
            face_encodings_pca = pca.transform(face_encodings)
            
            for face_encoding_pca in face_encodings_pca:
                distances = np.linalg.norm(known_face_encodings_pca - face_encoding_pca, axis=1)
                best_match_index = np.argmin(distances)
                min_distance = distances[best_match_index]
                
                if min_distance <= TOLERANCE:
                    confidence = 1.0 - (min_distance / TOLERANCE)
                    
                    if confidence > person_tracker.best_face_confidence:
                        person_tracker.best_face_confidence = confidence
                        person_tracker.best_face_frame = person_crop.copy()
                    
                    threshold = 0.4 if is_tracking else 0.5
                    
                    if confidence > threshold:
                        result["name"] = known_face_names[best_match_index]
                        result["confidence"] = confidence
                        
                        if is_tracking:
                            print(f"🔄 Tracked: {result['name']} (conf: {result['confidence']:.3f})")
                        else:
                            print(f"✅ Recognized: {result['name']} (conf: {result['confidence']:.3f})")
                        
                        break
        
    except Exception as e:
        if not is_tracking:
            print(f"Face recognition error: {e}")
    
    return result

# ========== FACE TRACKING MANAGEMENT ==========
def should_track_face_fast(person_tracker, current_time):
    time_since_last = current_time - person_tracker.last_face_track
    has_good_quality = person_tracker.current_quality >= FACE_QUALITY_THRESHOLD
    not_too_many_tracks = person_tracker.face_tracking_count < 30
    
    return (time_since_last >= FACE_TRACKING_INTERVAL and 
            has_good_quality and 
            not_too_many_tracks)

def handle_face_tracking_result(future, person_tracker):
    try:
        result = future.result(timeout=3.0)
        
        if result["confidence"] > 0.5:
            person_tracker.name = result["name"]
        
        append_fast_csv(face_csv, [
            result["timestamp"], person_tracker.track_id, 
            result["name"], result["confidence"], result["quality"]
        ])
        
    except Exception as e:
        pass

# ========== VEHICLE SAVING ==========
def save_vehicle_fast(vehicle_tracker):
    if vehicle_tracker.track_id in saved_vehicle_ids:
        return
    
    vehicle_path = ""
    if vehicle_tracker.best_frame is not None:
        vehicle_filename = f"vehicle_{vehicle_tracker.track_id}_{vehicle_tracker.vehicle_type}.jpg"
        vehicle_path = os.path.join(SAVE_DIR, "vehicles", vehicle_filename)
        cv2.imwrite(vehicle_path, vehicle_tracker.best_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    append_fast_csv(vehicle_csv, [
        now_ts(), vehicle_tracker.track_id, 
        vehicle_tracker.vehicle_type, vehicle_tracker.color, vehicle_path
    ])
    
    saved_vehicle_ids.add(vehicle_tracker.track_id)
    vehicle_tracker.saved = True
    print(f"💾 Saved {vehicle_tracker.vehicle_type} ID {vehicle_tracker.track_id}")

# ========== FAST CLEANUP ==========
def fast_cleanup(current_time):
    TIMEOUT = 6.0
    
    # Clean vehicles
    for tid in list(active_vehicles.keys()):
        if current_time - active_vehicles[tid].last_seen > TIMEOUT:
            if not active_vehicles[tid].saved:
                save_vehicle_fast(active_vehicles[tid])
            del active_vehicles[tid]
    
    # Clean persons - SAVE BEST IMAGE BEFORE DELETING
    for tid in list(active_persons.keys()):
        if current_time - active_persons[tid].last_seen > TIMEOUT:
            # Save the best person image before removing
            if not active_persons[tid].best_frame_saved and active_persons[tid].best_frame is not None:
                save_best_person_image(active_persons[tid])
            del active_persons[tid]

# ========== MAIN PROCESSING LOOP ==========
def main_processing_loop():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return
        
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎥 Video: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames")
    print(f"⚡ Processing: Every {DETECTION_SKIP + 1} frame(s)")
    
    small_w = PROCESSING_WIDTH
    small_h = int((PROCESSING_WIDTH / orig_w) * orig_h)
    
    if SHOW_VIDEO:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, VIDEO_WIDTH, int(VIDEO_WIDTH * orig_h / orig_w))
    
    frame_id = 0
    processed_frames = 0
    start_time = time.time()
    last_fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    total_face_tracks = 0
    
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            print("✅ Video ended")
            break
            
        frame_id += 1
        
        if frame_id % (DETECTION_SKIP + 1) != 0:
            continue
            
        processed_frames += 1
        current_time = time.time()
        
        fps_counter += 1
        if current_time - last_fps_time >= 1.0:
            current_fps = fps_counter / (current_time - last_fps_time)
            fps_counter = 0
            last_fps_time = current_time
        
        display_frame = frame.copy() if SHOW_VIDEO else None
        small_frame = cv2.resize(frame, (small_w, small_h))
        
        try:
            results = yolo.track(
                source=small_frame,
                imgsz=640,
                device=device,
                conf=0.4,
                iou=0.5,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                classes=[0, 2, 3, 5, 7],
                max_det=10
            )
            
            if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
                continue
                
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int) if hasattr(boxes, 'id') and boxes.id is not None else np.zeros(len(xyxy), dtype=int)
            
        except Exception as e:
            print(f"Detection error: {e}")
            continue
        
        for box, cls_id, conf, track_id in zip(xyxy, classes, confs, track_ids):
            if conf < 0.4:
                continue
                
            x1, y1, x2, y2 = fast_map_coords(box, small_w, small_h, orig_w, orig_h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w-1, x2), min(orig_h-1, y2)
            
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                continue
                
            hr_crop = frame[y1:y2, x1:x2].copy()
            if hr_crop is None or hr_crop.size == 0:
                continue
            
            # PERSON PROCESSING - SAVE BEST IMAGE
            if cls_id == 0 and area >= MIN_PERSON_AREA:
                if track_id not in active_persons:
                    active_persons[track_id] = FastPersonTracker(track_id)
                    print(f"👤 New person: ID {track_id}")
                
                tracker = active_persons[track_id]
                tracker.last_seen = current_time
                tracker.current_frame = hr_crop.copy()
                tracker.current_quality = fast_sharpness(hr_crop)
                
                # UPDATE BEST FRAME - Always track the best quality image
                if tracker.current_quality > tracker.best_quality:
                    tracker.best_quality = tracker.current_quality
                    tracker.best_frame = hr_crop.copy()
                    
                    # AUTO-SAVE if quality is really good
                    if tracker.best_quality > 25 and not tracker.best_frame_saved:
                        save_best_person_image(tracker)
                
                # FACE TRACKING
                if should_track_face_fast(tracker, current_time):
                    tracker.last_face_track = current_time
                    tracker.face_tracking_count += 1
                    total_face_tracks += 1
                    
                    future = face_executor.submit(ultra_fast_face_recognition, tracker.current_frame.copy(), tracker, True)
                    future.add_done_callback(lambda f: handle_face_tracking_result(f, tracker))
                
                # Try to save person with face recognition if good confidence
                if (not tracker.saved and tracker.best_face_confidence > 0.6 and 
                    current_time - tracker.first_seen > 2.0):
                    save_best_person_image(tracker)
                
                # Draw on display
                if SHOW_VIDEO:
                    color = get_fast_color(track_id)
                    name_display = tracker.name if tracker.name != "Unknown" else ""
                    draw_fast_detection(display_frame, x1, y1, x2, y2, track_id, "Person", conf, color, name_display)
            
            # VEHICLE PROCESSING - UNCHANGED
            elif cls_id in [2, 3, 5, 7] and area >= MIN_VEHICLE_AREA:
                if track_id not in active_vehicles:
                    active_vehicles[track_id] = FastVehicleTracker(track_id)
                
                tracker = active_vehicles[track_id]
                tracker.last_seen = current_time
                
                vehicle_types = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
                tracker.vehicle_type = vehicle_types.get(cls_id, "vehicle")
                
                quality = fast_sharpness(hr_crop)
                if quality > tracker.best_quality:
                    tracker.best_quality = quality
                    tracker.best_frame = hr_crop.copy()
                    tracker.color = get_dominant_color_fast(hr_crop)
                
                if (not tracker.saved and tracker.best_quality >= MIN_VEHICLE_QUALITY and 
                    current_time - tracker.first_seen > MIN_TRACK_DURATION):
                    save_vehicle_fast(tracker)
                
                if SHOW_VIDEO:
                    color = get_fast_color(track_id)
                    label = f"{tracker.vehicle_type}({tracker.color})"
                    draw_fast_detection(display_frame, x1, y1, x2, y2, track_id, label, conf, color)
        
        # Display
        if SHOW_VIDEO and display_frame is not None:
            add_fast_status_bar(display_frame, frame_id, total_frames, current_fps, 
                              len(saved_vehicle_ids), len(saved_person_ids), total_face_tracks)
            
            display_h = int(VIDEO_WIDTH * orig_h / orig_w)
            display_frame_resized = cv2.resize(display_frame, (VIDEO_WIDTH, display_h))
            cv2.imshow(WINDOW_NAME, display_frame_resized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup and auto-save persons
        if processed_frames % 15 == 0:
            fast_cleanup(current_time)
            
            # Also try to save any persons with good quality images that haven't been saved
            for tid, person in active_persons.items():
                if not person.best_frame_saved and person.best_quality >= PERSON_SAVE_QUALITY:
                    save_best_person_image(person)
        
        # Progress
        if processed_frames % 25 == 0:
            elapsed = time.time() - start_time
            fps_actual = processed_frames / elapsed
            progress = (frame_id / total_frames) * 100
            
            active_tracking = sum(1 for p in active_persons.values() if p.face_tracking_count > 0)
            
            print(f"📊 Frame {frame_id}/{total_frames} ({progress:.1f}%) | "
                  f"FPS: {fps_actual:.1f} | "
                  f"Vehicles: {len(saved_vehicle_ids)} | Persons: {len(saved_person_ids)} | "
                  f"Face Tracks: {total_face_tracks} | Active: {active_tracking}")
    
    # Final cleanup - SAVE ALL REMAINING PERSONS
    print("\n🔄 Final cleanup - saving all persons...")
    for tid, person in active_persons.items():
        if not person.best_frame_saved and person.best_frame is not None:
            save_best_person_image(person)
    
    for tracker in active_vehicles.values():
        if not tracker.saved:
            save_vehicle_fast(tracker)
    
    cap.release()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
    
    # Summary
    print("\n🎉 PROCESSING COMPLETED!")
    print(f"Total frames: {frame_id}")
    print(f"Processed: {processed_frames}")
    print(f"Vehicles saved: {len(saved_vehicle_ids)}")
    print(f"Persons saved: {len(saved_person_ids)}")
    print(f"Total face tracks: {total_face_tracks}")
    
    if saved_person_ids:
        print(f"\n✅ Successfully saved {len(saved_person_ids)} persons with best quality images!")

if __name__ == "__main__":
    print("🚀 STARTING FIXED PERSON SAVING PIPELINE")
    print(f"Device: {device}")
    print(f"Person save quality: >{PERSON_SAVE_QUALITY}")
    
    try:
        main_processing_loop()
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        face_executor.shutdown(wait=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Shutdown complete")