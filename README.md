# Perception-Track.ai
AI-powered perception system using YOLOv8 for real-time pose and vehicle detection. Detects lying-down poses with timed alerts via Email &amp; Telegram, and tracks vehicles and humans with color, type, and face recognition. Designed for safety monitoring and intelligent surveillance.

Maybe the file may seem a bit to long but if u Read every line and understand then it will be way easier to work with.
Try to do some changes and with imporvisation so that u will know what u r doing.

Overview:
Two AI-powered detection systems using YOLOv8:

1. Pose Detection - Detects lying down poses and sends alerts via Email and Telegram after 10s, 25s, 40s.
2. Vehicle and Human Recognition - Tracks vehicles (with color/type) and recognizes faces.

------------------------------------------------------------

Requirements:
- Python: 3.10.3 (Important)
- Hardware: Minimum 8GB RAM, GPU recommended
- Storage: At least 5GB free space

------------------------------------------------------------

Quick Setup:

1. Install Python 3.10.3
Download it from python.org and install it. Make sure to check "Add to PATH" during installation.

2. Clone and Setup Environment
Commands:
    git clone https://github.com/yourusername/perception-track-ai.git
    cd perception-track-ai
    python -m venv venv

Activate Virtual Environment:
    Windows: venv\Scripts\activate
    Linux/Mac: source venv/bin/activate

Install dependencies:
    pip install -r requirements.txt

3. Create Folder Structure:
    mkdir -p models dataset face_encodings pose_analysis_output/alerts
    mkdir -p up_optimized_m/vehicles up_optimized_m/persons up_optimized_m/best_faces

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

------------------------------------------------------------

Configuration:

A. Telegram Bot Setup (Pose Detection):
1. Open Telegram and search for BotFather.
2. Create a new bot and copy the bot token.
3. Get your chat ID by visiting:
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates

In code, update:
    BOT_TOKEN = "your_bot_token_here"
    CHAT_ID = "your_chat_id_here"

B. Gmail Setup (Pose Detection):
1. Enable 2-Factor Authentication in your Google Account.
2. Go to App Passwords, select Mail and Other, and generate a 16-character password.

In code, update:
    SENDER_ACCOUNTS = [
        {
            "email": "your_email@gmail.com",
            "password": "xxxx xxxx xxxx xxxx"
        }
    ]
    RECEIVER_EMAILS = ["recipient@gmail.com"]

C. Video Path Configuration:
    VIDEO_PATH = r"D:\path\to\your\video.mp4"

D. Face Recognition (Optional for Vehicle Pipeline):
If you don't have an encoding file, run:
    import pickle, numpy as np
    from sklearn.decomposition import PCA
    data = {"encodings": np.array([]), "names": [], "pca": PCA(n_components=50)}
    with open("face_encodings/face_encodings_pca.pkl", "wb") as f:
        pickle.dump(data, f)

------------------------------------------------------------

How to Run:

1. Pose Detection System:
    python "Pose with Notification Alert.py"

What Happens:
- Displays video window with skeleton overlay.
- Detects standing, sitting, and lying down.
- Sends alerts at 10s, 25s, and 40s of lying down.
- Press 'q' to quit or 'p' to pause.

Outputs:
- CSV file: pose_analysis_output/pose_analysis.csv
- Images: pose_analysis_output/alerts/

2. Vehicle and Human Pipeline:
    python "Vehical and human pipeline.py"

What Happens:
- Detects and tracks vehicles (car, motorcycle, bus, truck).
- Identifies dominant vehicle color.
- Tracks people with IDs and attempts face recognition.
- Saves best-quality frame for each detected object.

Outputs:
- Vehicles: up_optimized_m/vehicles/
- Persons: up_optimized_m/persons/
- Logs: up_optimized_m/*.csv

------------------------------------------------------------

Technical Explanation:

Pose Detection Logic:
- Standing: Torso angle <= 40 degrees.
- Sitting: Knees bent < 145 degrees.
- Lying Down: Torso angle >= 55 degrees (triggers alerts).

The system confirms pose over 3 frames before alerting.
Alerts are triggered at 10s, 25s, and 40s intervals.

Vehicle and Human Tracking:
- Detects vehicle type and extracts color from image.
- Tracks individuals using unique IDs.
- Saves high-quality images if detection confidence > 15.

------------------------------------------------------------

Common Issues and Fixes:

1. Module not found:
    pip install --upgrade pip
    pip install -r requirements.txt

2. Dlib installation errors:
    Linux:
        sudo apt-get install build-essential cmake libopenblas-dev
        pip install dlib
    Windows:
        Install Visual Studio Build Tools first.

3. CUDA not detected:
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

4. Telegram issues:
    - Verify bot token and chat ID.
    - Send a message to the bot before running script.

5. Email issues:
    - Use App Password, not regular Gmail password.

6. Video not found:
    - Use absolute path with 'r' prefix.
    - Verify file exists.

7. Slow processing:
    - Reduce processing width to 480.
    - Increase detection skip to 6.
    - Use GPU if available.

------------------------------------------------------------

CSV Output Formats:

pose_analysis.csv:
    timestamp, datetime, pose, person_id, confidence, torso_angle, knee_angle, hip_angle, long_term_alert, alert_duration, alert_count

vehicles.csv:
    timestamp, track_id, vehicle_type, color, image_path

persons.csv:
    timestamp, track_id, name, best_confidence, image_path, quality

------------------------------------------------------------

Best Practices:
1. Use clear, stable, 720p+ videos.
2. Ensure the subject is visible and not obstructed.
3. Use frontal faces for accurate recognition.
4. GPU acceleration improves speed 3-5x.

------------------------------------------------------------

Key Dependencies:
- YOLOv8 (Ultralytics) for object detection
- OpenCV for video processing
- face_recognition for facial analysis
- PyTorch as the deep learning backend
- aiohttp for fast async communication

------------------------------------------------------------

Credits:
- YOLOv8 by Ultralytics
- Face Recognition by Adam Geitgey
- Python 3.10.3

------------------------------------------------------------

End of Document.

