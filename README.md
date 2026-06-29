# Video Similarity Engine 🎥🔍

`video-similarity-engine` is an advanced Python-based video similarity and copy detection system. It combines low-level multimedia hashing with high-level AI context representation to create highly accurate, tamper-resistant video fingerprints. The engine is robust against transcoding, rotation, cropping, compression, and logo insertion.

The system natively supports **NVIDIA GPU (CUDA)** hardware acceleration for high-speed frame decoding via FFmpeg and parallel model inference via CLIP.

---

## ✨ Features (Multi-Feature Alignment)

Our comparison engine matches videos across 6 distinct, configurable layers:

1. **Blake3 Binary Hash**: 
   Millisecond-level exact binary matching for identical files.
2. **Perceptual Hashing (pHash & dHash)**: 
   Extracts frame-by-frame visual hashes to detect compression, resolution changes, or watermarks regardless of file size.
3. **Semantic Visual AI (OpenCLIP ViT-B-32)**: 
   Leverages CLIP deep learning embeddings to evaluate high-level visual context. Highly resistant to visual modifications, edits, and crop changes using parallel GPU *cosine similarity* calculations.
4. **Audio Chromaprint**: 
   Generates audio fingerprints via Chromaprint (Acoustid) to recognize matches with identical audio tracks, even when the visual stream is completely altered.
5. **Motion Hasher (Optical Flow)**: 
   Analyzes object motion patterns using *Gunner Farneback Optical Flow*. It registers motion magnitude and direction angle per second to index actions.
6. **Scene Detector (Structural Layout)**: 
   Identifies automatic video scene cuts to match the underlying editing structure and rhythm.

---

## 🛠️ System Requirements & Dependencies

This project requires Python 3.10+ and the following system libraries:
* **FFmpeg** (configured with CUDA/Nvidia headers if utilizing hardware acceleration)
* **Libchromaprint** (for audio fingerprinting extraction)

### System Package Installation (Ubuntu/Linux):
```bash
sudo apt update
sudo apt install ffmpeg libchromaprint-dev -y
```

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/DevZynk/video-similarity-engine.git
cd video-similarity-engine

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Adjust extraction parameters, match thresholds, and server options in `config.yml`. You can toggle features or change weights dynamically.

### 3. Start the API Server
Start the FastAPI development server:
```bash
python main.py
```
The server will be available at `http://localhost:8000`. You can access the interactive Swagger API documentation at `http://localhost:8000/docs`.

---

## 📡 API Endpoints

### 1. Extract & Save Fingerprint (`POST /fingerprint`)
Extracts all enabled features from a video and saves them locally, indexed by the video's Blake3 hash.

* **URL Parameter**: `path` (can be a local absolute file path or a direct video download URL)
* **Example Request**:
  `POST http://localhost:8000/fingerprint?path=/root/videos/video_sample.mp4`
* **Example Response**:
  ```json
  {
    "status": "success",
    "message": "Sidik jari video berhasil diekstrak dan disimpan.",
    "blake3_hash": "fa5594c06cb72f9135367b7a1b9f810ac7945fd25b38e2212d66bbc32ebb3ac4",
    "saved_path": "/root/project/df-videos/fingerprints_db/fa5594c06cb72f9135367b7a1b9f810ac7945fd25b38e2212d66bbc32ebb3ac4.json",
    "metadata": {
      "format": {
        "duration": "136.342000",
        "size": "143172725"
      }
    }
  }
  ```

---

### 2. Retrieve Fingerprint Details (`GET /fingerprint/{blake_hash}`)
Fetches cached metadata and frame-by-frame fingerprints from local storage.

* **Example Request**:
  `GET http://localhost:8000/fingerprint/fa5594c06cb72f9135367b7a1b9f810ac7945fd25b38e2212d66bbc32ebb3ac4`
* **JSON Output Structure**:
  All features are stored in database-ready structures containing explicit `frame_index` and `second` timestamps:
  ```json
  {
    "status": "success",
    "blake": "fa5594...",
    "phash": [
      { "frame_index": 1, "second": 1.0, "phash": "83b9...", "dhash": "5a96..." }
    ],
    "ai": {
      "ai_embeddings": [
        { "frame_index": 1, "second": 1.0, "embedding": [0.0008, 0.1739, ...] }
      ]
    },
    "motion": {
      "motion_fingerprints": [
        { "frame_index": 1, "second": 1.0, "magnitude": 1.5, "angle": 0.5 }
      ]
    }
  }
  ```

---

### 3. Compare Two Videos (`POST /compare`)
Compares two videos sequentially using active pipeline features.

* **URL Parameters**:
  * `video_source`: Blake3 Hash (pre-extracted), local file path, or download URL.
  * `video_target`: Same as above.
* **Example Request**:
  `POST http://localhost:8000/compare?video_source=fa5594...&video_target=/root/videos/test_clip.mp4`
* **Example Response (Duplicate Detected)**:
  ```json
  {
    "status": "duplicate",
    "match_type": "ai_embedding",
    "match_percentage": 100.0,
    "message": "Kecocokan semantik visual (AI) tinggi (100.0% >= threshold 90.0%).",
    "matched_segments": [
      {
        "video_1_second": 1.0,
        "video_2_second": 1.0,
        "video_1_frame_index": 1,
        "video_2_frame_index": 1,
        "cosine_similarity": 1.0
      }
    ]
  }
  ```

---

## ⚙️ Timeline Alignment & Matching Logic
Our comparison algorithms use a **Sliding Window** approach. If the target video has a different length (e.g., a 15-second clip cropped from a 2-minute original video), the target video's timeline is shifted frame-by-frame across the source video to identify the region with the highest correlation. The response returns the precise matched segment, listing exactly matched seconds and frame indices.

---

## 🧪 Running Tests
To execute the unit test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📝 License
This project is licensed under the MIT License. Feel free to use it for personal or commercial projects.
