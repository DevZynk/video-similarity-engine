from lib.temp_file import delete_temp_file
from lib.scene_detector import detect_video_scenes
from lib.motion_hasher import generate_video_motion_fingerprint
from lib.ai_embedding import generate_video_ai_embeddings
from lib.metadata import get_video_metadata
from lib.audio import generate_audio_fingerprint
from lib.phash import generate_phash_fingerprint
from lib.blake3 import blake3_file_fingerprint
from lib.temp_file import temp_file
from lib.config import config
from typing import Any, Dict, Literal

def video_fingerprint_per_second(data: Any, type: Literal["url", "blob", "file"]) -> Dict[str, Any]:
    """
    Extracts all fingerprint features of a video based on the active configuration.
    """
    file_path = None
    try:
        file_path = temp_file(type, data=data)
        
        # Extract all enabled features in config.yml
        blake = blake3_file_fingerprint(file_path) if config["features"]["blake3"]["enabled"] else None
        metadata = get_video_metadata(file_path)
        
        phash = generate_phash_fingerprint(file_path) if config["features"]["phash"]["enabled"] else []
        
        audio = (
            generate_audio_fingerprint(file_path) 
            if config["features"]["audio"]["enabled"] 
            else {"video_path": file_path, "duration": 0, "audio_subprints": []}
        )
        
        ai = (
            generate_video_ai_embeddings(file_path) 
            if config["features"]["ai_embedding"]["enabled"] 
            else {"video_path": file_path, "ai_embeddings": []}
        )
        
        motion = (
            generate_video_motion_fingerprint(file_path) 
            if config["features"]["motion"]["enabled"] 
            else {"video_path": file_path, "motion_fingerprints": []}
        )
        
        scene = (
            detect_video_scenes(file_path) 
            if config["features"]["scene"]["enabled"] 
            else {"video_path": file_path, "total_scenes": 0, "scenes": []}
        )
        
        return {
            "status": "success",
            "metadata": metadata,
            "blake": blake,
            "phash": phash,
            "audio": audio,
            "ai": ai,
            "motion": motion,
            "scene": scene
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        if file_path:
            delete_temp_file(file_path)