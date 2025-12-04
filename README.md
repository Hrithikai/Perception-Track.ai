# Perception-Track.ai
AI-powered perception system using YOLOv8 for real-time pose and vehicle detection. Detects lying-down poses with timed alerts via Email &amp; Telegram, and tracks vehicles and humans with color, type, and face recognition. Designed for safety monitoring and intelligent surveillance.

Maybe the file may seem a bit to long but if u Read every line and understand then it will be way easier to work with.
Try to do some changes and with improvisation of exsiting code.


## THIS SINGLE PROJECT CAN BE USED AS MULTIPLE PROJECTS ( SEPERATE THE SECTIONS )
Overview:
Two AI-powered detection systems using YOLOv8:

1. Pose Detection - Detects lying down poses and sends alerts via Email and Telegram after 10s, 25s, 40s.
2. Vehicle and Human Recognition - Tracks vehicles (with color/type) and recognizes faces.

## 💻 System Requirements

| Component | Requirement |
|------------|--------------|
| **Python** | 3.10.3 (Mandatory) |
| **RAM** | Minimum 8GB (16GB to 24GB RAM recommended)|
| **GPU** | Minimum 4GB VRAM Card Dedicated GPU (6GB or more recommended) required for real-time processing ,integrated gpu may not handle the load !!@!! |
| **CPU** | Minimum 4 cores and 8 threads(recommended 8 cores and 16 threads) |
| **Storage** | Minimum 6GB to 8GB free space |


---

## ⚙️ Installation Guide

### 1️⃣ Install Python
Download and install [Python 3.10.3](https://www.python.org/downloads/release/python-3103/)  
✔️ *Check “Add to PATH” during installation*

### 2️⃣ Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/yourusername/perception-track-ai.git
cd perception-track-ai

# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate       # Windows
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Folder Structure
```
perception-track-ai/
├── Pose with Notification Alert.py
├── Vehical and human pipeline.py
├── requirements.txt
├── models/                    # YOLO models (auto-downloaded)
├── dataset/                   # Place your videos here
├── face_encodings/            # Face recognition data
├── pose_analysis_output/      # Pose detection results
└── up_optimized_m/            # Vehicle/human output data
```

Create directories manually if missing:
```bash
mkdir -p models dataset face_encodings pose_analysis_output/alerts
mkdir -p up_optimized_m/{vehicles,persons,best_faces}
```

---

## 🤖 Configuration Steps

### 📨 Telegram Bot Setup (Pose Detection Alerts)
1. Open Telegram and start chat with **@BotFather**
2. Run `/newbot` → follow prompts → copy **Bot Token**
3. Get **Chat ID**:
   - Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Send message to bot, refresh to find `"chat":{"id":...}`

In code:
```python
BOT_TOKEN = "your_bot_token_here"
CHAT_ID = "your_chat_id_here"
```

### 📧 Gmail Setup (Email Alerts)
1. Enable **2-Step Verification** in Google Account  
2. Go to **App Passwords** → Generate 16-character password  
3. Replace credentials in code:
```python
SENDER_ACCOUNTS = [{
    "email": "your_email@gmail.com",
    "password": "xxxx xxxx xxxx xxxx"
}]
RECEIVER_EMAILS = ["recipient@gmail.com"]
```

### 🎥 Video Path Setup
```python
VIDEO_PATH = r"D:\path\to\your\video.mp4"
```

### 🧬 Optional: Face Recognition Encoding
If none exist, create empty encoding file:
```python
import pickle, numpy as np
from sklearn.decomposition import PCA

data = {"encodings": np.array([]), "names": [], "pca": PCA(n_components=50)}
with open("face_encodings/face_encodings_pca.pkl", "wb") as f:
    pickle.dump(data, f)
```

---

## ▶️ Running the Modules

### 🧍 Pose Detection System
```bash
python "Pose with Notification Alert.py"
```
**Functions:**
- Real-time pose classification: Standing / Sitting / Lying  
- Triggers alerts after 10s, 25s, and 40s of lying down  
- Alerts via **Email & Telegram**  
- Press `q` to exit or `p` to pause

**Outputs:**
- CSV: `pose_analysis_output/pose_analysis.csv`  
- Alerts: `pose_analysis_output/alerts/`

---

### 🚗 Vehicle & Human Detection
```bash
python "Vehical and human pipeline.py"
```
**Functions:**
- Tracks vehicles by type (car, bus, truck, motorcycle)  
- Detects vehicle color  
- Tracks humans with unique IDs  
- Saves best-quality frames and logs

**Outputs:**
- Vehicles: `up_optimized_m/vehicles/`  
- Persons: `up_optimized_m/persons/`  
- Logs: `up_optimized_m/*.csv`



### 🧍 Main.py where u find all features in one
```bash
python "Main.py"
```
**Functions:**
- Real-time pose classification: Standing / Sitting / Lying  
- Triggers alerts after 10s, 25s, and 40s of lying down  
- Alerts via **Email & Telegram**  
- Press `q` to exit or `p` to pause
- Tracks vehicles by type (car, bus, truck, motorcycle)  
- Detects vehicle color  
- Tracks humans with unique IDs  
- Saves best-quality frames and logs

**Outputs:**
- CSV: `pose_analysis_output/pose_analysis.csv`  
- Alerts: `pose_analysis_output/alerts/`
- Vehicles: `up_optimized_m/vehicles/`  
- Persons: `up_optimized_m/persons/`  
- Logs: `up_optimized_m/*.csv`

---

## 🧩 Detection Logic Summary

| Pose | Condition |
|------|------------|
| Standing | Torso angle ≤ 40° |
| Sitting | Knees bent < 145° |
| Lying | Torso angle ≥ 55° |

Pose detection triggers alerts at **10s, 25s, 40s**.  
Vehicle detection extracts **color**, **type**, and **best frame** based on quality score.

---

## 🧰 Troubleshooting

| Issue | Solution |
|--------|-----------|
| `ModuleNotFoundError` | pip install -r requirements.txt |
| dlib fails | Install CMake or Build Tools (Windows) |
| CUDA not detected | pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 |
| Telegram not working | Verify Token + Chat ID, message bot first |
| Email not sending | Use 16-char App Password |
| Slow performance | Reduce frame size or increase skip rate |

---

## 📊 Output File Formats

**pose_analysis.csv**
```
timestamp, datetime, pose, person_id, confidence, torso_angle, knee_angle, hip_angle, long_term_alert, alert_duration, alert_count
```

**vehicles.csv**
```
timestamp, track_id, vehicle_type, color, image_path
```

**persons.csv**
```
timestamp, track_id, name, best_confidence, image_path, quality
```

---

## ⚡ Optimization Tips
- Use GPU for 3–5x faster inference  
- Record stable, well-lit 720p+ videos  
- Avoid obstructions or extreme angles  
- Use frontal faces for accurate recognition  

---

## 🧠 Core Libraries
- **YOLOv8** (ultralytics) – Object Detection  
- **OpenCV** – Video Processing  
- **face_recognition** – Face Detection  
- **PyTorch** – Deep Learning Backend  
- **aiohttp** – Async Communication  

---

## 👨‍💻 Credits
- YOLOv8 by [Ultralytics](https://github.com/ultralytics)  
- Face Recognition by [Adam Geitgey](https://github.com/ageitgey/face_recognition)  
- Developed using **Python 3.10.3**

---

## 🏁 Summary
**Perception Track AI** combines pose and vehicle detection for advanced real-time situational awareness.  
It bridges the gap between **human activity analysis** and **vehicle intelligence**, ideal for smart cities, surveillance, and safety monitoring.

**Happy Tracking! 🚀**


