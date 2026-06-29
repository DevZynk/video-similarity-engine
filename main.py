from fastapi import FastAPI, HTTPException
import uvicorn
import os
import json
from core.finggerprint import video_fingerprint_per_second
from core.compare import compare_video
from lib.config import config

app = FastAPI(
    title="Video Fingerprinting API",
    description="API for extracting video fingerprints (pHash, AI, Audio, Motion, Scene) and analyzing duplication.",
    version="1.0.0"
)

# Folder for storing video fingerprints persistently based on config.yml
db_dir_config = config["server"]["db_dir"]
if os.path.isabs(db_dir_config):
    DB_DIR = db_dir_config
else:
    DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_dir_config)
os.makedirs(DB_DIR, exist_ok=True)


@app.post("/fingerprint", summary="Extract and store video fingerprint")
def fingerprint_video(path: str):
    """
    Extract all video features enabled in config.yml
    and save the result locally indexed by the video's Blake3 hash.
    """
    v_type = "url" if path.startswith(("http://", "https://")) else "file"
    
    if v_type == "file" and not os.path.exists(path):
        raise HTTPException(
            status_code=404, 
            detail=f"Video file not found at path: {path}"
        )
    
    result = video_fingerprint_per_second(path, v_type)
    if result.get("status") == "error":
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process video: {result.get('message')}"
        )
    
    # Save fingerprint result to JSON file using blake3 hash
    blake_hash = result.get("blake")
    save_path = os.path.join(DB_DIR, f"{blake_hash}.json")
    try:
        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to save fingerprint data to storage: {str(e)}"
        )
        
    return {
        "status": "success",
        "message": "Video fingerprint successfully extracted and saved.",
        "blake3_hash": blake_hash,
        "saved_path": save_path,
        "metadata": result.get("metadata")
    }


@app.get("/fingerprint/{blake_hash}", summary="Retrieve video fingerprint by Blake3 hash")
def get_fingerprint(blake_hash: str):
    """
    Retrieve detail video fingerprint data from local storage.
    """
    save_path = os.path.join(DB_DIR, f"{blake_hash}.json")
    if not os.path.exists(save_path):
        raise HTTPException(
            status_code=404, 
            detail="Fingerprint not found for the given hash."
        )
    
    try:
        with open(save_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to read fingerprint file: {str(e)}"
        )


@app.post("/compare", summary="Compare two videos to detect duplication")
def compare_videos(video_source: str, video_target: str):
    """
    Compares two videos. Input parameters can be:
    - **Blake3 Hash** (from previously extracted video fingerprint)
    - **Local File Path** (will be extracted on-the-fly)
    - **Video URL** (will be downloaded and extracted on-the-fly)
    
    The comparison runs sequentially based on enabled features in config.yml.
    Returned data shows matching seconds and method details.
    """
    # 1. Get fingerprint of the first video (video_source)
    v1 = None
    source_json_path = os.path.join(DB_DIR, f"{video_source}.json")
    if os.path.exists(source_json_path):
        try:
            with open(source_json_path, "r") as f:
                v1 = json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to read source data: {str(e)}"
            )
    else:
        v_type = "url" if video_source.startswith(("http://", "https://")) else "file"
        if v_type == "file" and not os.path.exists(video_source):
            raise HTTPException(
                status_code=404, 
                detail=f"video_source file/hash not found: {video_source}"
            )
        
        fp_res = video_fingerprint_per_second(video_source, v_type)
        if fp_res.get("status") == "error":
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process video_source: {fp_res.get('message')}"
            )
        v1 = fp_res
        
        # Save fingerprint for subsequent processing
        try:
            with open(os.path.join(DB_DIR, f"{v1['blake']}.json"), "w") as f:
                json.dump(v1, f, indent=4)
        except Exception:
            pass

    # 2. Get fingerprint of the second video (video_target)
    v2 = None
    target_json_path = os.path.join(DB_DIR, f"{video_target}.json")
    if os.path.exists(target_json_path):
        try:
            with open(target_json_path, "r") as f:
                v2 = json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to read target data: {str(e)}"
            )
    else:
        v_type = "url" if video_target.startswith(("http://", "https://")) else "file"
        if v_type == "file" and not os.path.exists(video_target):
            raise HTTPException(
                status_code=404, 
                detail=f"video_target file/hash not found: {video_target}"
            )
        
        fp_res = video_fingerprint_per_second(video_target, v_type)
        if fp_res.get("status") == "error":
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process video_target: {fp_res.get('message')}"
            )
        v2 = fp_res
        
        # Save fingerprint for subsequent processing
        try:
            with open(os.path.join(DB_DIR, f"{v2['blake']}.json"), "w") as f:
                json.dump(v2, f, indent=4)
        except Exception:
            pass

    # 3. Run sequential comparison logic
    try:
        compare_res = compare_video(v1, v2)
        return compare_res
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to compare videos: {str(e)}"
        )


if __name__ == "__main__":
    # Use host and port from config.yml configuration file
    host_ip = os.getenv("HOST", config["server"]["host"])
    port_num = int(os.getenv("PORT", config["server"]["port"]))
    uvicorn.run(app, host=host_ip, port=port_num)
